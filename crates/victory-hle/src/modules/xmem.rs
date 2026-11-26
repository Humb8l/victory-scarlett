//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let (asked_kind, asked_protect) = (kind, protect);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let allocate = |parameter: &mut MEM_EXTENDED_PARAMETER| unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    let mut at = allocate(&mut parameter);
    // A 32-bit request that found no room below 4 GiB: the guards there
    // make way for it.
    if at.is_null() && constrained && highest < FOUR_GIB {
        at = low_guards::retry(size, || allocate(&mut parameter));
    }
    if base.is_null() {
        // The census keeps what the title asked for (`VICTORY_CENSUS`).
        #[cfg(windows)]
        victory_gpu::title_memory_reserved(at as usize, size, asked_kind, flags, asked_protect);
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    // Committed: what the GPU side found uncommitted there is no longer so
    // (it looks again only now and then otherwise).
    if !at.is_null() && kind & MEM_COMMIT != 0 {
        victory_gpu::title_memory_committed(at as usize, size);
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // (F10, `VICTORY_MAP_DEFER`) What the title asked for so far, as it asked.
    DEFER.before(address as usize, 1);
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// A section, its size in bytes, and (run-46, `VICTORY_DSTORAGE_LINEAR_VIEW`)
/// one view of all of it of our own, made the first time DirectStorage
/// writes into it (0: none could be made), with the pages committed through
/// it, a bit each.
/// (Storage wait job, `VICTORY_POOL_DISCARD`) and the pages written through
/// that view and not discarded since, a bit each; (fault drivers) the pages
/// discarded and not written since, a bit each.
struct Section(
    HANDLE,
    usize,
    std::sync::OnceLock<usize>,
    std::sync::OnceLock<Box<[std::sync::atomic::AtomicU64]>>,
    std::sync::OnceLock<Box<[std::sync::atomic::AtomicU64]>>,
    std::sync::OnceLock<Box<[std::sync::atomic::AtomicU64]>>,
);

/// (Fault drivers) Counts a DirectStorage write of `len` bytes from `within`
/// bytes into a page: its bytes and the 4 KiB pages it touches.
fn count_disk(within: usize, len: usize) {
    use victory_gpu::faults::{Driver, count};
    if len == 0 {
        return;
    }
    count(Driver::DiskBytes, len as u64);
    count(Driver::DiskPages, ((within + len).div_ceil(4096) - within / 4096) as u64);
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        if let Some(&at) = self.2.get()
            && at != 0
        {
            use windows_sys::Win32::System::Memory::UnmapViewOfFile;
            // SAFETY: our view of all of the section, nothing refers to it any
            // more (no slice of the section is left).
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }) };
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.0) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// (Storage wait job) `Physical::pages` again, read-mostly: page number ->
/// slice changes only when the title allocates or frees physical pages, and
/// is written here under the page table's lock at the same two places. The
/// DirectStorage workers look their pages up here ([`runs`]) instead of in
/// the page table, whose lock every mapping call holds through its host
/// calls (run-49: 349 ms of worker time a loading second spent waiting).
static PAGE_SLICES: LazyLock<std::sync::RwLock<HashMap<usize, Slice>>> = LazyLock::new(Default::default);

fn page_slices() -> std::sync::RwLockReadGuard<'static, HashMap<usize, Slice>> {
    PAGE_SLICES.read().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    let section = Arc::new(Section(handle, size, std::sync::OnceLock::new(), std::sync::OnceLock::new(), std::sync::OnceLock::new(), std::sync::OnceLock::new()));
    let mut physical = physical();
    let mut mirror = PAGE_SLICES.write().unwrap_or_else(|e| e.into_inner());
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        mirror.insert(number, slice.clone());
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    let mut mirror = PAGE_SLICES.write().unwrap_or_else(|e| e.into_inner());
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        let page = unsafe { *pages.add(i) };
        mirror.remove(&page);
        physical.pages.remove(&page);
    }
    1
}

/// (Run-46) Whether DirectStorage writes physical pages through one lasting
/// view of each whole section (`VICTORY_DSTORAGE_LINEAR_VIEW`, default on;
/// `=0`: a view made and unmapped for each request, as before). A fresh view
/// takes a soft fault on the first touch of each of its 4 KiB pages (16 per
/// 64 KiB page: 8,192 pages are 131,072 faults) and its unmapping flushes
/// every core's TLB; the lasting view faults once per page for good and is
/// never unmapped while the section lives. It does not depend on where the
/// title maps the pages (its CPU and GPU views), so a deferred change of
/// those views never meets the layer's own writes.
fn linear_view() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_DSTORAGE_LINEAR_VIEW").is_none_or(|v| v != "0");
        tracing::info!(on, "VICTORY_DSTORAGE_LINEAR_VIEW: DirectStorage writes physical pages through one lasting view of each section");
        on
    })
}

/// The address of `slice` (and the `n - 1` slices after it) in the lasting
/// view of its section, committed; `None` if the view cannot be made or the
/// commit fails (the caller maps a view of its own, as before).
fn linear_at(slice: &Slice, n: usize) -> Option<*mut u8> {
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
    let section = &*slice.section;
    let base = *section.2.get_or_init(|| {
        // SAFETY: a view of all of a section we own, kept until it is dropped.
        let at = unsafe { MapViewOfFile(section.0, FILE_MAP_WRITE, 0, 0, 0) }.Value as usize;
        if at == 0 {
            tracing::warn!(bytes = section.1, error = %std::io::Error::last_os_error(), "dstorage: no lasting view of a section (VICTORY_DSTORAGE_LINEAR_VIEW); a view per request");
        }
        at
    });
    if base == 0 || (slice.index + n) * PAGE > section.1 {
        return None;
    }
    let bits = section.3.get_or_init(|| (0..(section.1 / PAGE).div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect());
    let all = |k: usize| bits[k / 64].load(Relaxed) & (1 << (k % 64)) != 0;
    let at = base + slice.index * PAGE;
    if !(slice.index..slice.index + n).all(all) {
        if !commit(at, n * PAGE) {
            return None;
        }
        for k in slice.index..slice.index + n {
            bits[k / 64].fetch_or(1 << (k % 64), Relaxed);
        }
    }
    // (`VICTORY_POOL_DISCARD`) Resident through this view from now on.
    let resident = section.4.get_or_init(|| (0..(section.1 / PAGE).div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect());
    for k in slice.index..slice.index + n {
        if resident[k / 64].fetch_or(1 << (k % 64), Relaxed) & (1 << (k % 64)) == 0 {
            POOL[P_RESIDENT].fetch_add(1, Relaxed);
        }
    }
    // (Fault drivers) A page discarded since its last write: this write
    // takes fresh zero-filled pages (counted as all 16 of its 4 KiB pages).
    if let Some(discarded) = section.5.get() {
        for k in slice.index..slice.index + n {
            if discarded[k / 64].fetch_and(!(1 << (k % 64)), Relaxed) & (1 << (k % 64)) != 0 {
                victory_gpu::faults::count(victory_gpu::faults::Driver::DiskAfterDiscard, (PAGE / 4096) as u64);
                discard_rewritten((Arc::as_ptr(&slice.section) as usize, k));
            }
        }
    }
    Some(at as *mut u8)
}

/// (Storage wait job) `VICTORY_POOL_DISCARD` (default 1; 0: off). The lasting
/// view of a section (`VICTORY_DSTORAGE_LINEAR_VIEW`) keeps every page it ever
/// wrote in this process's working set, also after the title has given the
/// page back: run-49 peaked at 24.7 GB working set against run-47's 19.1, the
/// title's pool being 6.1 GiB. The title's streaming pool is one allocation
/// it shows whole at one address for good (its pool view, `[0x1471bfaa8]`)
/// and maps page by page into its CPU and GPU streaming views from a free
/// list (executor 0x140f2c470: kinds 0-2 take a page off the list, 3-5 put it
/// back); a page shown by no view but a whole-section one is on that free
/// list and its bytes are dead (the console reuses it as it is; whoever takes
/// it next maps it and has DirectStorage or its own code write what it
/// reads). Such a page, still so 2 s after it was last unmapped, is reset in
/// the lasting view (`MEM_RESET`: not written to the pagefile) and taken out
/// of the working set (`VirtualUnlock`), under the page table's lock, so no
/// map of it can come between the check and the reset. Pages of sections
/// without a whole view are never touched (their bytes outlive a mapping).
fn pool_discard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = linear_view() && std::env::var_os("VICTORY_POOL_DISCARD").is_none_or(|v| v != "0");
        tracing::info!(on, "VICTORY_POOL_DISCARD: pool pages the title has given back (shown by no view but its whole pool view, 2 s on) are reset and taken out of the working set in the layer's lasting view");
        on
    })
}

/// Pages resident through the lasting views, pages discarded since the start,
/// discard calls the host refused.
static POOL: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
const P_RESIDENT: usize = 0;
const P_DISCARDED: usize = 1;
const P_FAILED: usize = 2;

/// Pages that lost a view, to look at again later: when, and the page.
static UNSHOWN: Mutex<std::collections::VecDeque<(std::time::Instant, Slice)>> = Mutex::new(std::collections::VecDeque::new());

/// Whether only whole-section views show the page `key` now, and at least one
/// does (the page is on the title's free list).
fn only_whole_views(physical: &Physical, key: (usize, usize), size: usize) -> bool {
    let Some(addresses) = physical.shown.get(&key) else {
        return false;
    };
    !addresses.is_empty()
        && addresses.iter().all(|&a| {
            physical
                .views
                .range(..=a)
                .next_back()
                .is_some_and(|(&at, (len, slice))| at + len > a && *len == size && slice.index == 0)
        })
}

/// Called by `unmap` for each page of a view it takes away: a page resident
/// through the lasting view that only the pool view shows now is noted.
fn note_unshown(physical: &Physical, slice: &Slice, k: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if !pool_discard() {
        return;
    }
    let section = &*slice.section;
    let index = slice.index + k;
    let Some(resident) = section.4.get() else {
        return;
    };
    if resident[index / 64].load(Relaxed) & (1 << (index % 64)) == 0 || !only_whole_views(physical, slice.key(k), section.1) {
        return;
    }
    let page = Slice { section: slice.section.clone(), index };
    UNSHOWN.lock().unwrap_or_else(|e| e.into_inner()).push_back((std::time::Instant::now(), page));
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-pool-discard".into()).spawn(discard_loop);
    });
}

/// (Fault drivers) Free RAM at or above which nothing is discarded:
/// `VICTORY_POOL_DISCARD_BELOW_MB` (4096; 0: always discard, as before).
fn discard_below() -> u64 {
    static MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let mb = *MB.get_or_init(|| {
        let mb = std::env::var("VICTORY_POOL_DISCARD_BELOW_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096);
        tracing::info!(mb, "VICTORY_POOL_DISCARD_BELOW_MB: pool pages are discarded only while free RAM is under this (0: always)");
        mb
    });
    if mb == 0 { u64::MAX } else { mb << 20 }
}

/// Free physical RAM in bytes.
fn free_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullAvailPhys)
}

/// Discarding is held back now because free RAM is plentiful.
static HELD_BY_RAM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The most pages kept noted while discarding is held back.
const UNSHOWN_MOST: usize = 1 << 18;

/// (Prefault job) When each page was discarded, until it is written again
/// (or 5 s on): to count pages discarded and then rewritten within 5 s
/// (thrashing: a page given back and taken again soon).
static DISCARDED_AT: LazyLock<Mutex<HashMap<(usize, usize), std::time::Instant>>> = LazyLock::new(Default::default);

/// Discarded pages (64 KiB) in the current adaptive window, and of them
/// rewritten within 5 s.
static WINDOW: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A discarded page written again: counted if within 5 s of its discard.
fn discard_rewritten(key: (usize, usize)) {
    use std::sync::atomic::Ordering::Relaxed;
    let at = DISCARDED_AT.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
    if at.is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(5)) {
        victory_gpu::faults::count(victory_gpu::faults::Driver::DiscardRewritten5s, (PAGE / 4096) as u64);
        WINDOW[1].fetch_add(1, Relaxed);
    }
}

/// (Prefault job) The wait after a page's last unmapping before it is
/// discarded: `VICTORY_POOL_DISCARD_DELAY_S` (default 0: adaptive, 2 s to
/// 10 s; a number: that many seconds, 2 as before).
fn discard_delay_setting() -> u64 {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let s = std::env::var("VICTORY_POOL_DISCARD_DELAY_S").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
        tracing::info!(s, "VICTORY_POOL_DISCARD_DELAY_S: wait after a pool page's last unmapping before it is discarded (0: adaptive 2-10 s by pages rewritten within 5 s)");
        s
    })
}

/// The adaptive delay in ms (`VICTORY_POOL_DISCARD_DELAY_S=0`).
static DELAY_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(2000);

/// The next delay from the last window's discards `d` and of them rewritten
/// within 5 s `r`, and free RAM: more than a quarter rewritten doubles it
/// (up to 10 s), under 5 % halves it (down to 2 s); under 1 GiB free RAM it
/// is 2 s.
fn next_delay(ms: u64, d: u64, r: u64, free: Option<u64>) -> u64 {
    if free.is_some_and(|f| f < 1 << 30) {
        return 2000;
    }
    if d >= 64 && r * 4 > d {
        (ms * 2).min(10_000)
    } else if d >= 64 && r * 20 < d {
        (ms / 2).max(2000)
    } else {
        ms
    }
}

/// The wait in force now.
fn discard_delay() -> std::time::Duration {
    match discard_delay_setting() {
        0 => std::time::Duration::from_millis(DELAY_MS.load(std::sync::atomic::Ordering::Relaxed)),
        s => std::time::Duration::from_secs(s),
    }
}

/// The discarding thread: twice a second, the pages noted [`discard_delay`]
/// ago or more (2 s before the prefault job).
fn discard_loop() {
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{MEM_RESET, VirtualUnlock};
    let mut window = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        // (Prefault job) The adaptive delay, every 5 s; stale notes of
        // discard times dropped.
        if window.elapsed() >= std::time::Duration::from_secs(5) {
            window = std::time::Instant::now();
            let (d, r) = (WINDOW[0].swap(0, Relaxed), WINDOW[1].swap(0, Relaxed));
            let ms = next_delay(DELAY_MS.load(Relaxed), d, r, free_ram());
            DELAY_MS.store(ms, Relaxed);
            victory_gpu::faults::DISCARD_DELAY_MS.store(discard_delay().as_millis() as u64, Relaxed);
            DISCARDED_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, at| at.elapsed() < std::time::Duration::from_secs(5));
        }
        let quiet = discard_delay();
        // (Fault drivers) Free RAM is the only reason to discard: with more
        // than `VICTORY_POOL_DISCARD_BELOW_MB` free nothing is discarded (a
        // discarded page costs fresh zero-fill faults at its next write);
        // the pages stay noted (the oldest dropped past a cap) for when RAM
        // runs short.
        if free_ram().is_some_and(|free| free >= discard_below()) {
            HELD_BY_RAM.store(true, Relaxed);
            let mut queue = UNSHOWN.lock().unwrap_or_else(|e| e.into_inner());
            let over = queue.len().saturating_sub(UNSHOWN_MOST);
            queue.drain(..over);
            continue;
        }
        HELD_BY_RAM.store(false, Relaxed);
        let due: Vec<Slice> = {
            let mut queue = UNSHOWN.lock().unwrap_or_else(|e| e.into_inner());
            let n = queue.iter().take_while(|(at, _)| at.elapsed() >= quiet).count();
            queue.drain(..n).map(|(_, s)| s).collect()
        };
        // Pieces of 1,024 pages, the lock let go between them.
        for piece in due.chunks(1024) {
            let physical = physical();
            // (lasting view base, first index, pages)
            let mut runs: Vec<(usize, usize, usize)> = Vec::new();
            for page in piece {
                let section = &*page.section;
                let (Some(&base), Some(resident)) = (section.2.get(), section.4.get()) else {
                    continue;
                };
                let bit = 1u64 << (page.index % 64);
                if base == 0 || resident[page.index / 64].load(Relaxed) & bit == 0 || !only_whole_views(&physical, page.key(0), section.1) {
                    continue;
                }
                resident[page.index / 64].fetch_and(!bit, Relaxed);
                DISCARDED_AT.lock().unwrap_or_else(|e| e.into_inner()).insert(page.key(0), std::time::Instant::now());
                WINDOW[0].fetch_add(1, Relaxed);
                let discarded = section.5.get_or_init(|| (0..(section.1 / PAGE).div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect());
                discarded[page.index / 64].fetch_or(bit, Relaxed);
                match runs.last_mut() {
                    Some((b, first, n)) if *b == base && *first + *n == page.index => *n += 1,
                    _ => runs.push((base, page.index, 1)),
                }
            }
            for (base, first, n) in runs {
                let (at, len) = ((base + first * PAGE) as *mut c_void, n * PAGE);
                // SAFETY: committed pages of our lasting view whose bytes are
                // dead (above); the lock keeps maps of them out meanwhile.
                let reset = unsafe { VirtualAlloc(at, len, MEM_RESET, PAGE_READWRITE) };
                if reset.is_null() && POOL[P_FAILED].fetch_add(1, Relaxed) == 0 {
                    tracing::warn!(error = %std::io::Error::last_os_error(), "VICTORY_POOL_DISCARD: MEM_RESET refused (taken out of the working set only)");
                }
                // SAFETY: as above; on pages not locked this only takes them
                // out of the working set (it reports "not locked").
                unsafe { VirtualUnlock(at, len) };
                POOL[P_RESIDENT].fetch_sub(n as u64, Relaxed);
                POOL[P_DISCARDED].fetch_add(n as u64, Relaxed);
                victory_gpu::faults::count(victory_gpu::faults::Driver::PoolDiscarded, (n * PAGE / 4096) as u64);
            }
            drop(physical);
        }
    }
}

/// The housekeeping line's part: pool pages resident through the lasting
/// views, discarded this run, and the process's working set.
fn pool_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mut c = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    let mb = |pages: u64| (pages * PAGE as u64) >> 20;
    let failed = POOL[P_FAILED].load(Relaxed);
    format!(
        "; pool pages resident {} MB, discarded this run {} MB, working set {} MB (peak {} MB) (VICTORY_POOL_DISCARD={}{}; only under {} MB free RAM, VICTORY_POOL_DISCARD_BELOW_MB{}); mapping calls that asked no commit this run {} (VICTORY_COMMIT_ONCE={})",
        mb(POOL[P_RESIDENT].load(Relaxed)),
        mb(POOL[P_DISCARDED].load(Relaxed)),
        c.WorkingSetSize >> 20,
        c.PeakWorkingSetSize >> 20,
        u8::from(pool_discard()),
        if failed == 0 { String::new() } else { format!(", resets refused {failed}") },
        if discard_below() == u64::MAX { "any".to_owned() } else { (discard_below() >> 20).to_string() },
        if HELD_BY_RAM.load(Relaxed) { "; held now: RAM plentiful" } else { "" },
        COMMITS_SKIPPED.load(Relaxed),
        u8::from(commit_once())
    )
}

/// (Run-46) The DirectStorage workers' time in the page table's lock and in
/// making / unmapping views of their own, ns (the worker split).
pub(crate) static WORKER_NS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
pub(crate) const W_LOCK: usize = 0;
pub(crate) const W_VIEW: usize = 1;
pub(crate) const W_UNMAP: usize = 2;
/// (Storage wait job) Page lookups after the lock ([`runs`]), copies into
/// scattered pages ([`write_physical`]), and the page-table lookup of the
/// notes ([`pages_written`]: [`shown_at`] with its lock wait; part of the
/// notes bucket, said apart).
pub(crate) const W_LOOKUP: usize = 3;
pub(crate) const W_COPY: usize = 4;
pub(crate) const W_NOTE_LOOKUP: usize = 5;

fn worker_add(what: usize, since: std::time::Instant) {
    WORKER_NS[what].fetch_add(since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.0,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        commit(at.Value as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        let t = std::time::Instant::now();
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
        worker_add(W_UNMAP, t);
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let t = std::time::Instant::now();
    let physical = page_slices();
    worker_add(W_LOCK, t);
    let t = std::time::Instant::now();
    let found = runs_in(&physical, pages);
    worker_add(W_LOOKUP, t);
    found
}

fn runs_in(physical: &HashMap<usize, Slice>, pages: &[usize]) -> Option<Vec<(Slice, usize)>> {
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let k = (n * PAGE - within).min(data.len() - done);
        count_disk(within, k);
        let t = std::time::Instant::now();
        if let Some(at) = linear_view().then(|| linear_at(slice, *n)).flatten() {
            worker_add(W_VIEW, t);
            let t = std::time::Instant::now();
            // SAFETY: `n` committed pages of the lasting view; the bytes are
            // the request's alone until it completes.
            unsafe { std::ptr::copy_nonoverlapping(data[done..].as_ptr(), at.add(within), k) };
            worker_add(W_COPY, t);
        } else {
            let Some(mut view) = View::map(slice, *n) else {
                return false;
            };
            worker_add(W_VIEW, t);
            let t = std::time::Instant::now();
            view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
            worker_add(W_COPY, t);
        }
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    /// The view made for it (none in the lasting view of the section).
    _view: Option<View>,
    /// The section of the lasting view, kept while the window lives.
    _keep: Option<Slice>,
    at: *mut u8,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: `len` committed writable bytes of a view that lives as long
        // as `self` (its own, or the section's lasting one, which lives while
        // a slice of the section does: the page table holds them while the
        // request runs).
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    let t = std::time::Instant::now();
    if let Some(at) = linear_view().then(|| linear_at(slice, *n)).flatten() {
        worker_add(W_VIEW, t);
        count_disk(offset % PAGE, len);
        // SAFETY: inside the `n` pages.
        return Some(PageWindow { _view: None, _keep: Some(slice.clone()), at: unsafe { at.add(offset % PAGE) }, len });
    }
    let mut view = View::map(slice, *n)?;
    worker_add(W_VIEW, t);
    count_disk(offset % PAGE, len);
    // SAFETY: inside the view.
    let at = unsafe { view.bytes().as_mut_ptr().add(offset % PAGE) };
    Some(PageWindow { _view: Some(view), _keep: None, at, len })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            // (F10, `VICTORY_MAP_DEFER`) Not where an unmapping is pending:
            // the page is not shown there any more as far as the title knows.
            pieces.extend(addresses.iter().filter(|&&at| !DEFER.pending_at(at)).map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    let t = std::time::Instant::now();
    let shown = shown_at(pages, offset, len);
    worker_add(W_NOTE_LOOKUP, t);
    let prefault = super::prefault::on();
    for (at, bytes) in shown {
        victory_gpu::title_memory_written(at, bytes, true);
        // (`VICTORY_MAP_PREFAULT`) Written now: resident, read through the
        // title's views off its threads.
        if prefault {
            super::prefault::queued(super::prefault::Ask { at, len: bytes });
        }
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let _faults = victory_gpu::faults::scope(victory_gpu::faults::Kind::Map);
    if !pages.is_null() {
        victory_gpu::faults::count(victory_gpu::faults::Driver::CpuViewPages, (n * PAGE / 4096) as u64);
    }
    // (F10, `VICTORY_MAP_DEFER`) Pending unmappings of these pages first, in
    // call order (none on the thread applying them).
    DEFER.before(base, n.saturating_mul(PAGE));
    let mut physical = timed(victory_gpu::mapfast::Split::LockWait, physical);
    // (Neptune, `VICTORY_REMAP_SAME`) The pages are there already, each where it is asked
    // for, or nothing is there and nothing is asked for: on the console that costs the
    // title next to nothing, and it asks all the time (in Neptune's profile a thread of
    // the title stood in these calls for 84% of its time, a view taken away and one made
    // for every 64 KiB page). Mapped again, every address would show the same bytes of
    // the same section as now, committed as now: no view is taken away or made, and the
    // notes of a remapping (placed textures and their host copies let go, guards
    // dropped, views forgotten) are not made, as nothing they describe changed; the
    // protection the guards rely on stays as it is, with them. The GPU side is told the
    // bytes may have changed (`kernel_will_write`: guarded blocks opened, placed textures
    // there looked at again), as a remapping told it (`=2`: it is told nothing; `=0`:
    // mapped again, as before).
    let mode = remap_same();
    if mode != 0 {
        // SAFETY: the title passes `n` page numbers, or null.
        let asked = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) });
        let same = match asked {
            Some(asked) => shows(&physical, base, asked),
            None => shows_nothing(&physical, base, n * PAGE),
        };
        same_count(n, asked.is_some(), same);
        if same {
            drop(physical);
            if mode == 1 && asked.is_some() {
                timed(victory_gpu::mapfast::Split::Notes, || victory_gpu::kernel_will_write(base, n * PAGE));
            }
            return address;
        }
    }
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`); reads of other
    // bytes go on.
    let (from, to) = reach(&physical, base, n * PAGE);
    let _changing = timed(victory_gpu::mapfast::Split::Hold, || victory_gpu::title_memory_remapping(from, to - from));
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    let _faults = victory_gpu::faults::scope(victory_gpu::faults::Kind::Map);
    if !pages.is_null() {
        // SAFETY: the title passes `n` page numbers (0: unmap that address).
        let mapped = (0..n).filter(|&i| unsafe { *pages.add(i) } != 0).count();
        victory_gpu::faults::count(victory_gpu::faults::Driver::CpuViewPages, (mapped * PAGE / 4096) as u64);
    }
    // (F3) Timed as `map_alias` is (the split with `VICTORY_LAZY_GPU_VIEW=1`),
    // counted always (the frame log's "cpu view").
    let timing = victory_gpu::mapfast::timing();
    let started = std::time::Instant::now();
    TIMED.with(|t| t.set(if timing { TIMED_CPU } else { 0 }));
    let (ok, runs) = if super::mapdefer::on() {
        // (F10, `VICTORY_MAP_DEFER`) Unmappings deferred, maps page by page.
        scatter_deferred(addresses, n, pages)
    } else if victory_gpu::mapfast::on_cpu() {
        scatter_runs(addresses, n, pages)
    } else {
        (scatter_pages(addresses, n, pages), n as u64)
    };
    TIMED.with(|t| t.set(0));
    victory_gpu::mapfast::cpu_call(n as u64, runs, started.elapsed().as_nanos() as u64);
    if !timing {
        cpu_view_line();
    }
    BOOL::from(ok)
}

/// (F3) The CPU view's line every five seconds while the meter (which says
/// it with the GPU view's) is off.
fn cpu_view_line() {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = std::time::Instant::now();
    if now.duration_since(*last.get_or_insert(now)).as_secs_f64() < 5.0 {
        return;
    }
    *last = Some(now);
    tracing::info!("{}", victory_gpu::mapfast::take_cpu_line().trim_start_matches(" | "));
}

/// Entry `i` of a scatter call: its address and its page (`None`: unmap).
///
/// # Safety
/// `addresses` holds `n > i` addresses and `pages`, unless null, `n` pages.
unsafe fn scatter_entry(addresses: *const *mut c_void, pages: *const usize, i: usize) -> (usize, Option<usize>) {
    // SAFETY: as the caller promises.
    unsafe {
        let page = if pages.is_null() { 0 } else { *pages.add(i) };
        (*addresses.add(i) as usize, (page != 0).then_some(page))
    }
}

/// (F10, `VICTORY_MAP_DEFER`) The deferred unmappings of both views.
pub(crate) static DEFER: LazyLock<super::mapdefer::Deferrer> =
    LazyLock::new(|| super::mapdefer::Deferrer::new(apply_deferred, super::mapdefer::LIMIT_PAGES));

/// (F10) Applies a deferred unmapping as the call would have: the CPU view
/// page by page (`XMemMapPhysicalPages(at, 1, null)`), the GPU view as
/// `map_alias(at, 0, len)` in pieces of at most 16 views cut at view starts
/// (no view is cut that the whole range would not cut; the page table's lock
/// is let go between pieces, so the frame thread's maps are not held up long).
fn apply_deferred(op: &super::mapdefer::Op) {
    use super::mapdefer::Op;
    match *op {
        Op::Cpu { at, pages } => {
            for k in 0..pages {
                XMemMapPhysicalPages((at + k * PAGE) as *mut c_void, 1, std::ptr::null());
            }
        }
        Op::Gpu { at, len } => {
            let end = at.saturating_add(len);
            let starts: Vec<usize> = if len == 0 {
                Vec::new()
            } else {
                physical().views.range(at + 1..end).map(|(a, _)| *a).collect()
            };
            let mut from = at;
            for (i, b) in starts.into_iter().enumerate() {
                if (i + 1) % 16 == 0 {
                    map_alias(from as u64, 0, (b - from) as u64);
                    from = b;
                }
            }
            map_alias(from as u64, 0, (end - from) as u64);
        }
    }
}

/// (F10, `VICTORY_MAP_DEFER`) `XMemMapPhysicalPagesScatter` with its
/// unmappings deferred (entries that follow on page by page as one
/// operation, applied page by page) and its maps done here page by page, as
/// with `VICTORY_MAP_FAST_CPU=0`; each map applies the pending unmappings of
/// its page first. Unmappings always succeed today; a failing map ends the
/// call as before.
fn scatter_deferred(addresses: *const *mut c_void, n: usize, pages: *const usize) -> (bool, u64) {
    use super::mapdefer::Op;
    let mut run: Option<(usize, usize)> = None;
    let flush = |run: &mut Option<(usize, usize)>| {
        if let Some((at, pages)) = run.take() {
            DEFER.defer(Op::Cpu { at, pages });
        }
    };
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (at, page) = unsafe { scatter_entry(addresses, pages, i) };
        match page {
            None => match &mut run {
                Some((first, k)) if first.checked_add(*k * PAGE) == Some(at) => *k += 1,
                _ => {
                    flush(&mut run);
                    run = Some((at, 1));
                }
            },
            Some(p) => {
                flush(&mut run);
                if XMemMapPhysicalPages(at as *mut c_void, 1, &p).is_null() {
                    return (false, i as u64 + 1);
                }
            }
        }
    }
    flush(&mut run);
    (true, n as u64)
}

/// `XMemMapPhysicalPagesScatter` page by page (`VICTORY_MAP_FAST=0`).
fn scatter_pages(addresses: *const *mut c_void, n: usize, pages: *const usize) -> bool {
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return false;
        }
    }
    true
}

/// (F3, `VICTORY_MAP_FAST`) `XMemMapPhysicalPagesScatter` by runs: entries
/// whose addresses follow one another page by page, all mapping or all
/// unmapping (up to `MOST_PAGES`), are one group ([`scatter_group`]); a
/// lone or unaligned entry goes page by page as before. Whether all went
/// well, and the runs it took.
fn scatter_runs(addresses: *const *mut c_void, n: usize, pages: *const usize) -> (bool, u64) {
    let most = victory_gpu::mapfast::MOST_PAGES as usize;
    // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
    let entry = |i: usize| unsafe { scatter_entry(addresses, pages, i) };
    let (mut i, mut runs) = (0, 0u64);
    while i < n {
        let (first, page) = entry(i);
        let mut j = i + 1;
        if first.is_multiple_of(PAGE) {
            while j < n && j - i < most {
                let (at, p) = entry(j);
                if Some(at) != first.checked_add((j - i) * PAGE) || p.is_some() != page.is_some() {
                    break;
                }
                j += 1;
            }
        }
        let ok = if j - i == 1 {
            runs += 1;
            !XMemMapPhysicalPages(first as *mut c_void, 1, page.as_ref().map_or(std::ptr::null(), |p| p as *const usize)).is_null()
        } else {
            let group: Option<Vec<usize>> = page.map(|_| (i..j).map(|k| entry(k).1.unwrap_or(0)).collect());
            match scatter_group(first, j - i, group.as_deref()) {
                Some(r) => {
                    runs += r;
                    true
                }
                None => false,
            }
        };
        if !ok {
            return (false, runs);
        }
        i = j;
    }
    (true, runs)
}

/// (F3, `VICTORY_MAP_FAST`) `count` pages at `base` on, mapped from `pages`
/// (one each) or, with `None`, unmapped: what `XMemMapPhysicalPages(base +
/// k * PAGE, 1, ..)` page after page leaves, with one lock, one hold and one
/// set of notes per run instead of per page. The runs it took; `None` if a
/// page failed (as the page by page call: the pages before it done).
///
/// Why the same: each page's "there already" (`VICTORY_REMAP_SAME`) depends
/// only on what its own address shows, which mapping or unmapping its
/// neighbours does not change (an unmap maps again what stays of a view
/// around the range, the same pages at the same addresses), so it is asked
/// up front. Pages there already are left alone and told as before
/// (`kernel_will_write` over the piece, a union of the pages' ranges); the
/// others are unmapped and mapped as runs of slices that follow one another
/// in one section (placeholders joined first), told by ranges, the union of
/// those page by page, in the same order relative to the host calls
/// (`unmap_with` fast). A run that fails to map as one, and everything after
/// it in the group, is done page by page.
fn scatter_group(base: usize, count: usize, pages: Option<&[usize]>) -> Option<u64> {
    use victory_gpu::mapfast::{Count, Split};
    let mode = remap_same();
    let mut physical = timed(Split::LockWait, physical);
    let page_by_page = |from: usize| -> Option<u64> {
        counted(Count::PageByPage, 1);
        for k in from..count {
            let page = pages.map(|p| &p[k] as *const usize).unwrap_or(std::ptr::null());
            if XMemMapPhysicalPages((base + k * PAGE) as *mut c_void, 1, page).is_null() {
                return None;
            }
        }
        Some((count - from) as u64)
    };
    let Some(pages) = pages else {
        // Unmapping: pieces each of pages showing nothing (nothing done) or
        // all showing something.
        let mut pieces: Vec<(usize, usize, bool)> = Vec::new();
        for k in 0..count {
            let nothing = mode != 0 && shows_nothing(&physical, base + k * PAGE, PAGE);
            match pieces.last_mut() {
                Some((_, n, was)) if *was == nothing => *n += 1,
                _ => pieces.push((k, 1, nothing)),
            }
        }
        let mut runs = 0;
        for (k, n, nothing) in pieces {
            if mode != 0 {
                same_count(n, false, nothing);
            }
            if nothing {
                continue;
            }
            let (at, len) = (base + k * PAGE, n * PAGE);
            let (start, end) = reach(&physical, at, len);
            let _changing = timed(Split::Hold, || victory_gpu::title_memory_remapping(start, end - start));
            unmap_with(&mut physical, at, len, true);
            runs += 1;
        }
        return Some(runs);
    };
    // Mapping: every page known first, or page by page (which says which).
    let slices: Option<Vec<Slice>> = pages.iter().map(|p| physical.pages.get(p).cloned()).collect();
    let Some(slices) = slices else {
        drop(physical);
        return page_by_page(0);
    };
    // Pieces: pages there already, or runs of slices that follow one another.
    let mut pieces: Vec<(usize, usize, bool)> = Vec::new();
    let (mut same_n, mut other_n) = (0, 0);
    for k in 0..count {
        let same = mode != 0 && slice_key(&physical, base + k * PAGE) == Some(slices[k].key(0));
        if same {
            same_n += 1;
        } else {
            other_n += 1;
        }
        match pieces.last_mut() {
            Some((at, n, was))
                if *was == same
                    && (same || (Arc::ptr_eq(&slices[*at].section, &slices[k].section) && slices[k].index == slices[*at].index + *n)) =>
            {
                *n += 1;
            }
            _ => pieces.push((k, 1, same)),
        }
    }
    if mode != 0 {
        same_count(same_n, true, true);
        same_count(other_n, true, false);
    }
    let mut runs = 0;
    for (k, n, same) in pieces {
        let (at, len) = (base + k * PAGE, n * PAGE);
        if same {
            if mode == 1 {
                timed(Split::Notes, || victory_gpu::kernel_will_write(at, len));
            }
            continue;
        }
        let (start, end) = reach(&physical, at, len);
        let changing = timed(Split::Hold, || victory_gpu::title_memory_remapping(start, end - start));
        unmap_with(&mut physical, at, len, true);
        let split = n == 1 || !join_placeholders(at, len);
        if split && n > 1 {
            // Not one placeholder: page by page from here (the unmap done
            // stays done; page by page finds nothing there to unmap).
            drop(changing);
            drop(physical);
            return page_by_page(k).map(|r| r + runs);
        }
        if !map_with(&mut physical, at, len, slices[k].clone(), split) {
            drop(changing);
            drop(physical);
            return page_by_page(k).map(|r| r + runs);
        }
        if n > 1 {
            counted(Count::Runs, 1);
            counted(Count::RunPages, n as u64);
        }
        runs += 1;
    }
    Some(runs)
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    use victory_gpu::mapfast::{Split, add};
    let _faults = victory_gpu::faults::scope(victory_gpu::faults::Kind::Map);
    if from != 0 {
        victory_gpu::faults::count(victory_gpu::faults::Driver::GpuViewPages, len / 4096);
    }
    // (F10, `VICTORY_MAP_DEFER`) An unmapping (always a success here)
    // deferred; a map applies the pending unmappings of its destination and
    // of its source first (what the source shows is what is copied).
    if from == 0 && super::mapdefer::on() {
        DEFER.defer(super::mapdefer::Op::Gpu { at: to as usize, len: len as usize });
        return true;
    }
    DEFER.before(to as usize, len as usize);
    if from != 0 {
        DEFER.before(from as usize, len as usize);
    }
    // (F3) The timing split, with the meter (`VICTORY_LAZY_GPU_VIEW=1`).
    let started = victory_gpu::mapfast::timing().then(std::time::Instant::now);
    if started.is_some() {
        victory_gpu::mapfast::gpu_pages((len as usize).div_ceil(PAGE) as u64);
    }
    TIMED.with(|t| t.set(if started.is_some() { TIMED_GPU } else { 0 }));
    let ok = map_alias_with(to as usize, from as usize, len as usize, victory_gpu::mapfast::on());
    TIMED.with(|t| t.set(0));
    if let Some(started) = started {
        add(Split::Alias, started.elapsed().as_nanos() as u64);
    }
    ok
}

thread_local! {
    /// (F3) Which of this thread's mapping calls is timed now ([`timed`]):
    /// 0 none, [`TIMED_GPU`] a `map_alias` call, [`TIMED_CPU`] an
    /// `XMemMapPhysicalPagesScatter` call.
    static TIMED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

const TIMED_GPU: u8 = 1;
const TIMED_CPU: u8 = 2;

/// The path the timed call on this thread is charged to.
#[inline]
fn timed_path() -> Option<victory_gpu::mapfast::Path> {
    match TIMED.with(std::cell::Cell::get) {
        TIMED_GPU => Some(victory_gpu::mapfast::Path::GpuView),
        TIMED_CPU => Some(victory_gpu::mapfast::Path::CpuView),
        _ => None,
    }
}

/// (F3) `f`, its time added to `split` while a mapping call is timed on
/// this thread.
#[inline]
fn timed<T>(split: victory_gpu::mapfast::Split, f: impl FnOnce() -> T) -> T {
    let Some(path) = timed_path() else {
        return f();
    };
    let t = std::time::Instant::now();
    let r = f();
    victory_gpu::mapfast::add_to(path, split, t.elapsed().as_nanos() as u64);
    r
}

/// (F3) Counted while a mapping call is timed on this thread.
#[inline]
fn counted(c: victory_gpu::mapfast::Count, n: u64) {
    if let Some(path) = timed_path() {
        victory_gpu::mapfast::count_to(path, c, n);
    }
}

/// [`map_alias`]; `fast`: `VICTORY_MAP_FAST`.
fn map_alias_with(to: usize, from: usize, len: usize, fast: bool) -> bool {
    use victory_gpu::mapfast::Split;
    let mut physical = timed(Split::LockWait, physical);
    // (Neptune) As in `XMemMapPhysicalPages` (`VICTORY_REMAP_SAME`): what is asked for is
    // there already, page for page.
    let mode = remap_same();
    if mode != 0 {
        let same_at = |physical: &Physical, k: usize| {
            let there = slice_key(physical, from + k);
            there.is_some() && there == slice_key(physical, to + k)
        };
        let (same, some) = if from == 0 {
            let same = shows_nothing(&physical, to, len);
            (same, same)
        } else if fast {
            // (`VICTORY_MAP_FAST`) A run asked for as one: its pages there already are
            // left as they are, page for page, as when asked for one by one.
            let n = (0..len).step_by(PAGE).filter(|&k| same_at(&physical, k)).count();
            (n == len.div_ceil(PAGE), n > 0)
        } else {
            let same = (0..len).step_by(PAGE).all(|k| same_at(&physical, k));
            (same, same)
        };
        if same {
            SAME[3].fetch_add((len / PAGE) as u64, std::sync::atomic::Ordering::Relaxed);
            // (Counted, and the line said, with `XMemMapPhysicalPages`'s.)
            same_count(0, false, false);
            SAME[4].fetch_add((len / PAGE) as u64, std::sync::atomic::Ordering::Relaxed);
            drop(physical);
            if mode == 1 && from != 0 {
                timed(Split::Notes, || victory_gpu::kernel_will_write(to, len));
            }
            return true;
        }
        if some {
            // Some of the pages are there already: the run in pieces, each all there
            // already or none of it, in order (each counted by its own call).
            let mut pieces: Vec<(usize, usize, bool)> = Vec::new();
            for k in (0..len).step_by(PAGE) {
                let here = same_at(&physical, k);
                let n = PAGE.min(len - k);
                match pieces.last_mut() {
                    Some((_, bytes, was)) if *was == here => *bytes += n,
                    _ => pieces.push((k, n, here)),
                }
            }
            drop(physical);
            return pieces.into_iter().fold(true, |ok, (k, n, _)| map_alias_with(to + k, from + k, n, fast) && ok);
        }
        SAME[3].fetch_add((len / PAGE) as u64, std::sync::atomic::Ordering::Relaxed);
        same_count(0, false, false);
    }
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let (start, end) = reach(&physical, to, len);
    let _changing = timed(Split::Hold, || victory_gpu::title_memory_remapping(start, end - start));
    unmap_with(&mut physical, to, len, fast);
    // (Neptune, `VICTORY_ALIAS_RUNS`) One view for each run of pages that follow one
    // another in their section, as `XMemMapPhysicalPages` makes them: page by page, a
    // texture level of 16 MiB was 256 views, each made by a call into the kernel and each
    // taken away by another when the level goes. Not where the two ranges overlap: there
    // each page is looked up after the ones before it are mapped, as before. (The source
    // pages are looked up after `unmap` of the destination, as one by one.)
    let apart = from.saturating_add(len) <= to || to.saturating_add(len) <= from;
    if from != 0 && apart && alias_runs() {
        let pages = len / PAGE;
        let mut k = 0;
        while k < pages {
            let Some(first) = slice_at(&physical, from + k * PAGE) else {
                return false;
            };
            let mut run = 1;
            while k + run < pages
                && slice_at(&physical, from + (k + run) * PAGE)
                    .is_some_and(|s| Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + run)
            {
                run += 1;
            }
            // A view lies in one placeholder: one cut into pages by an earlier mapping is
            // not one any more (placeholders are not joined again), and a view across
            // two fails (error 487). The run ends where the placeholder does.
            // (`VICTORY_MAP_FAST`) The placeholders under a run are joined into one
            // first, so that the run is one view; one page needs no look at its
            // placeholder (the run is one page whatever it says).
            let (run, split) = if !fast {
                (run.min(placeholder_pages(to + k * PAGE)).max(1), true)
            } else if run == 1 {
                (1, true)
            } else if join_placeholders(to + k * PAGE, run * PAGE) {
                (run, false)
            } else {
                (run.min(placeholder_pages(to + k * PAGE)).max(1), true)
            };
            if !map_with(&mut physical, to + k * PAGE, run * PAGE, first, split) {
                return false;
            }
            SAME[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            SAME[6].fetch_add(run as u64, std::sync::atomic::Ordering::Relaxed);
            k += run;
        }
        // A tail shorter than a page, mapped page by page below as before.
        if len % PAGE == 0 {
            return true;
        }
        let k = pages * PAGE;
        let Some(slice) = slice_at(&physical, from + k) else {
            return false;
        };
        return map(&mut physical, to + k, PAGE, slice);
    }
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// (Neptune) What a mapping of pages that are shown there already does
/// (`VICTORY_REMAP_SAME`): 1, the default: no view is taken away or made, and the GPU side
/// is told the bytes may have changed; 2: nothing; 0: the pages are mapped again.
fn remap_same() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("VICTORY_REMAP_SAME").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// (Neptune) Whether a copy of page mappings makes one view for a run of pages
/// (`VICTORY_ALIAS_RUNS=0`: one for every page, as before).
fn alias_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_RUNS").is_none_or(|v| v != "0"))
}

/// The whole pages from `at` to the end of the placeholder (or other region) that holds
/// it; 1 if the host cannot say.
fn placeholder_pages(at: usize) -> usize {
    // SAFETY: plain data the call fills in.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a query of our own process.
    let got = timed(victory_gpu::mapfast::Split::Placeholder, || unsafe {
        VirtualQueryEx(GetCurrentProcess(), at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>())
    });
    counted(victory_gpu::mapfast::Count::PlaceholderCalls, 1);
    if got == 0 {
        return 1;
    }
    let end = info.BaseAddress as usize + info.RegionSize;
    (end.saturating_sub(at) / PAGE).max(1)
}

/// The host's flag that joins placeholders that follow one another into one
/// (`VirtualFree`; not in windows-sys).
const MEM_COALESCE_PLACEHOLDERS: u32 = 1;

/// (`VICTORY_MAP_FAST`) Makes `[at, at + len)` exactly one placeholder: the
/// placeholders under it cut at its ends and joined. False if any of it is
/// not a placeholder or the host refuses; what it did by then leaves
/// placeholders (cut or not), which the caller's fallback maps page by
/// placeholder as before.
fn join_placeholders(at: usize, len: usize) -> bool {
    use victory_gpu::mapfast::{Count, Split};
    let end = at + len;
    // The placeholders under the range: each its own allocation.
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut p = at;
    while p < end {
        // SAFETY: plain data the call fills in.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a query of our own process.
        let got = timed(Split::Placeholder, || unsafe {
            VirtualQueryEx(GetCurrentProcess(), p as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>())
        });
        counted(Count::PlaceholderCalls, 1);
        let (start, stop) = (info.AllocationBase as usize, info.BaseAddress as usize + info.RegionSize);
        if got == 0 || info.State != MEM_RESERVE || start > p || stop <= p {
            return false;
        }
        pieces.push((start, stop));
        p = stop;
    }
    let (first, last) = (pieces[0], pieces[pieces.len() - 1]);
    if pieces.len() == 1 && first == (at, end) {
        return true;
    }
    let free = |base: usize, bytes: usize, how: u32| {
        counted(Count::PlaceholderCalls, 1);
        // SAFETY: acts on placeholders only (fails on anything else): cuts one
        // around a range inside it, or joins those that make up the range.
        timed(Split::Placeholder, || unsafe { VirtualFree(base as *mut c_void, bytes, MEM_RELEASE | how) }) != 0
    };
    // The ends cut: a placeholder reaching out at the start (and, if it is
    // the only one, at the end), and one reaching out at the end.
    if (first.0 < at || (pieces.len() == 1 && first.1 > end)) && !free(at, first.1.min(end) - at, MEM_PRESERVE_PLACEHOLDER) {
        return false;
    }
    if pieces.len() > 1 && last.1 > end && !free(last.0, end - last.0, MEM_PRESERVE_PLACEHOLDER) {
        return false;
    }
    pieces.len() == 1 || free(at, len, MEM_COALESCE_PLACEHOLDERS)
}

/// The slice a view shows at `at`, if one does.
fn slice_at(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (at - base) / PAGE,
            section: s.section.clone(),
        })
}

/// What a view shows at `at` (section and page index), if one does.
fn slice_key(physical: &Physical, at: usize) -> Option<(usize, usize)> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| s.key((at - base) / PAGE))
}

/// Whether `[base, base + pages)` shows `pages` already, page for page (none
/// unknown).
fn shows(physical: &Physical, base: usize, pages: &[usize]) -> bool {
    pages.iter().enumerate().all(|(i, page)| {
        let Some(slice) = physical.pages.get(page) else {
            return false;
        };
        slice_key(physical, base + i * PAGE) == Some(slice.key(0))
    })
}

/// Whether no view shows any of `[base, base + len)`.
fn shows_nothing(physical: &Physical, base: usize, len: usize) -> bool {
    let before = physical.views.range(..=base).next_back().is_some_and(|(at, (size, _))| at + size > base);
    !before && physical.views.range(base..base.saturating_add(len)).next().is_none()
}

/// (Neptune) Since the last line: pages asked to be mapped or unmapped, of them there
/// already, of nothing unmapped (nothing done), pages of copied mappings, of them there
/// already, and views made for runs of copied pages and their pages ([`same_count`]).
static SAME: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Counts `n` pages asked of [`XMemMapPhysicalPages`] (`mapped`: pages, not an unmapping;
/// `same`: there already), with a line every five seconds.
fn same_count(n: usize, mapped: bool, same: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    SAME[0].fetch_add(n as u64, Relaxed);
    if same {
        SAME[if mapped { 1 } else { 2 }].fetch_add(n as u64, Relaxed);
    }
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = std::time::Instant::now();
    let secs = now.duration_since(*last.get_or_insert(now)).as_secs_f64();
    if secs < 5.0 {
        return;
    }
    *last = Some(now);
    let w = [0, 1, 2, 3, 4, 5, 6].map(|i| SAME[i].swap(0, Relaxed) as f64 / secs);
    tracing::info!(
        "title memory mappings asked for, a second over {secs:.0} s: {:.0} pages mapped or unmapped ({:.0} of them there already, {:.0} of nothing unmapped: nothing done), {:.0} of copied mappings ({:.0} there already) (VICTORY_REMAP_SAME); {:.1} alias runs made, {:.0} pages in them (VICTORY_ALIAS_RUNS)",
        w[0], w[1], w[2], w[3], w[4], w[5], w[6]
    );
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    map_with(physical, base, len, slice, true)
}

/// [`map`]; `split`: false when the range is known to be one whole
/// placeholder (no cut needed).
fn map_with(physical: &mut Physical, base: usize, len: usize, slice: Slice, split: bool) -> bool {
    use victory_gpu::mapfast::{Count, Split};
    counted(Count::Views, 1);
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = timed(Split::MapOs, || unsafe {
        if split {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        MapViewOfFile3(
            slice.section.0,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    });
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    // (neptune-ve 62cda3d, `VICTORY_COMMIT_ONCE`) A commit belongs to the
    // section, not to the view it was made through (the author tried it:
    // pages committed through one view read as committed through another,
    // also one mapped later), and nothing here decommits section pages
    // (`MEM_RESET` discards keep the commit): pages committed before, through
    // any view, are not asked again.
    let pages = len.div_ceil(PAGE);
    let bits = commit_once().then(|| committed_bits(&slice.section));
    let known = bits.as_ref().is_some_and(|b| slice.index + pages <= b.len() * 64 && (slice.index..slice.index + pages).all(|k| b[k / 64].load(std::sync::atomic::Ordering::Relaxed) & (1 << (k % 64)) != 0));
    if known {
        COMMITS_SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        if !timed(Split::MapOs, || commit(base, len)) {
            // SAFETY: the view just mapped, back to a placeholder.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            return false;
        }
        if let Some(b) = bits {
            for k in slice.index..(slice.index + pages).min(b.len() * 64) {
                b[k / 64].fetch_or(1 << (k % 64), std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    timed(Split::Notes, || {
        // New contents, read-write: the GPU side's copies no longer guard it.
        victory_gpu::title_memory_remapped(base, len);
        // What the range shows now, for the GPU side's in-place census.
        victory_gpu::title_memory_view(base, len, slice.section.0 as usize, (slice.index * PAGE) as u64);
    });
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    // (`VICTORY_MAP_PREFAULT`) Its page-table entries over pages already
    // resident in the section filled in on the victory-prefault thread,
    // before the title or a reader touches them (never-written pages are
    // left alone: reading them would zero-fill them).
    if super::prefault::on() {
        prefault_resident(base, len, &slice);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// (`VICTORY_MAP_PREFAULT`) Queues the runs of `[base, base + len)` (a view
/// of `slice` on) whose 64 KiB pages are resident through the section's
/// lasting view.
fn prefault_resident(base: usize, len: usize, slice: &Slice) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(resident) = slice.section.4.get() else {
        return;
    };
    let pages = len.div_ceil(PAGE);
    let mut k = 0;
    while k < pages {
        let is = |k: usize| {
            let i = slice.index + k;
            resident.get(i / 64).is_some_and(|w| w.load(Relaxed) & (1 << (i % 64)) != 0)
        };
        if !is(k) {
            k += 1;
            continue;
        }
        let first = k;
        while k < pages && is(k) {
            k += 1;
        }
        let at = base + first * PAGE;
        super::prefault::queued(super::prefault::Ask { at, len: ((k - first) * PAGE).min(base + len - at) });
    }
}

/// (neptune-ve 62cda3d) Whether the title's mapping calls ask for the commit
/// only of section pages not committed before through any view
/// (`VICTORY_COMMIT_ONCE`, default 1 as the author's; 0: at every map).
fn commit_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !on_wine() && std::env::var_os("VICTORY_COMMIT_ONCE").is_none_or(|v| v != "0"))
}

/// Mapping calls that found their pages committed and asked nothing.
static COMMITS_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The section's committed pages, a bit each (shared with the lasting view,
/// `linear_at`).
fn committed_bits(section: &Section) -> &[std::sync::atomic::AtomicU64] {
    section.3.get_or_init(|| (0..(section.1 / PAGE).div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect())
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        victory_gpu::faults::count(victory_gpu::faults::Driver::Committed, (len / 4096) as u64);
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Start and end of the bytes whose mapping `unmap(physical, base, len)`
/// and mapping `[base, base + len)` again change: the range, and the whole
/// of the views reaching out of it (`unmap` takes them away and maps what
/// stays of them again). Under the page table's lock, as `unmap` is.
fn reach(physical: &Physical, base: usize, len: usize) -> (usize, usize) {
    let end = base.saturating_add(len);
    if len == 0 {
        return (base, end); // `unmap` changes nothing
    }
    let mut reach = (base, end);
    // As `unmap` finds them: the view holding `base`, the last starting
    // inside (views do not overlap).
    if let Some((&at, &(size, _))) = physical.views.range(..=base).next_back()
        && at + size > base
    {
        reach = (at, end.max(at + size));
    }
    if let Some((&at, &(size, _))) = physical.views.range(base + 1..end).next_back() {
        reach.1 = reach.1.max(at + size);
    }
    reach
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    unmap_with(physical, base, len, false)
}

/// [`unmap`]; `fast` (`VICTORY_MAP_FAST`, for copied mappings): the GPU side
/// is told once for each run of views that follow one another, not once per
/// view, in the same order relative to the host calls (what the views showed
/// is gone before they are unmapped and before what stays of them is mapped
/// again; what was really unmapped after). Its notes are by ranges, and the
/// union of ranges that touch is told the same as the ranges one by one.
fn unmap_with(physical: &mut Physical, base: usize, len: usize, fast: bool) {
    use victory_gpu::mapfast::{Count, Split};
    if fast && len != 0 {
        let end = base + len;
        let first = physical
            .views
            .range(..=base)
            .next_back()
            .filter(|(at, (size, _))| *at + size > base)
            .map(|(at, _)| *at);
        let hit: Vec<usize> = first.into_iter().chain(physical.views.range(base + 1..end).map(|(at, _)| *at)).collect();
        let mut gone: Vec<(usize, usize, Slice)> = Vec::with_capacity(hit.len());
        for at in hit {
            let Some((size, slice)) = physical.views.remove(&at) else {
                continue;
            };
            for k in 0..size.div_ceil(PAGE) {
                let key = slice.key(k);
                if let Some(addresses) = physical.shown.get_mut(&key) {
                    addresses.retain(|&a| a != at + k * PAGE);
                    if addresses.is_empty() {
                        physical.shown.remove(&key);
                    }
                }
                note_unshown(physical, &slice, k);
            }
            gone.push((at, size, slice));
        }
        // Runs of views that follow one another (views do not overlap; in address order).
        let runs = view_runs(gone.iter().map(|&(at, size, _)| (at, size)));
        timed(Split::Notes, || {
            for &(lo, hi) in &runs {
                victory_gpu::title_memory_remapped(lo, hi - lo);
            }
        });
        for (at, size, slice) in gone {
            counted(Count::UnmapCalls, 1);
            // SAFETY: a view we mapped; it becomes a placeholder again.
            timed(Split::UnmapOs, || unsafe {
                UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }, MEM_PRESERVE_PLACEHOLDER)
            });
            if at < base {
                counted(Count::Cut, 1);
                map(physical, at, base - at, slice.clone());
            }
            if at + size > end {
                counted(Count::Cut, 1);
                let index = slice.index + (end - at) / PAGE;
                map(physical, end, at + size - end, Slice { index, ..slice });
            }
        }
        timed(Split::Notes, || {
            for (lo, hi) in runs {
                let (lo, hi) = (lo.max(base), hi.min(end));
                if lo < hi {
                    victory_gpu::title_memory_unmapped(lo, hi - lo);
                }
            }
        });
        return;
    }
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
            note_unshown(physical, &slice, k);
        }
        timed(Split::Notes, || victory_gpu::title_memory_remapped(at, size));
        counted(Count::UnmapCalls, 1);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        timed(Split::UnmapOs, || unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        });
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
        // What was really taken away: the view's bytes inside the range (the
        // rest of it is mapped again just above, the same pages).
        let (lo, hi) = (at.max(base), (at + size).min(end));
        if lo < hi {
            timed(Split::Notes, || victory_gpu::title_memory_unmapped(lo, hi - lo));
        }
    }
}

/// (`VICTORY_MAP_FAST`) Views (address, bytes; in address order, none
/// overlapping) as runs (start, end) of views that follow one another.
fn view_runs(views: impl Iterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, size) in views {
        match runs.last_mut() {
            Some((_, stop)) if *stop == at => *stop = at + size,
            _ => runs.push((at, at + size)),
        }
    }
    runs
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    // (F10) The deferral's part of the "frame thread streaming" line.
    victory_gpu::framestream::add_to_line(super::mapdefer::take_line);
    // (Storage wait job) Pool residency on the housekeeping line.
    victory_gpu::resource::add_to_housekeeping(pool_line);
    // (Prefault job) The prefault meter on the "fault drivers" line.
    victory_gpu::faults::add_to_drivers_line(super::prefault::take_line);
    // (Hang watch) Every thread's stack after the watch's line.
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        victory_gpu::hangwatch::set_stacks(victory_loader::dump_stacks);
        victory_loader::set_symbolizer(victory_gpu::hangwatch::describe);
    }
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// (Neptune, `VICTORY_REMAP_SAME`, `VICTORY_ALIAS_RUNS`) Pages mapped again where they are
    /// shown already keep their view; a copied mapping of a run of pages is one view; and
    /// a mapping that changes anything is made as before.
    #[test]
    fn mapping_what_is_there_already_keeps_the_views() {
        let pages = allocate(3);
        let reserve = || XMemVirtualAlloc(std::ptr::null_mut(), 3 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().wrapping_add(2 * PAGE).write(7) };
        let views_at = |at: *mut c_void, len: usize| -> Vec<(usize, usize)> {
            let physical = physical();
            let at = at as usize;
            physical.views.range(at..at + len).map(|(&base, &(size, _))| (base, size)).collect()
        };
        let before = views_at(a, 3 * PAGE);
        assert_eq!(before, [(a as usize, 3 * PAGE)]);
        // The same pages again, all or some of them: nothing changes.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a.wrapping_byte_add(PAGE), 1, pages[1..].as_ptr()), a.wrapping_byte_add(PAGE));
        assert_eq!(views_at(a, 3 * PAGE), before);
        // Nothing unmapped where nothing is: nothing changes.
        assert_eq!(XMemMapPhysicalPages(b, 3, std::ptr::null()), b);
        assert!(views_at(b, 3 * PAGE).is_empty());
        // A copied mapping of three pages that follow one another: one view.
        assert!(map_alias(b as u64, a as u64, 3 * PAGE as u64));
        assert_eq!(views_at(b, 3 * PAGE), [(b as usize, 3 * PAGE)]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().wrapping_add(2 * PAGE).read() }, 7);
        // Copied again: there already.
        assert!(map_alias(b as u64, a as u64, 3 * PAGE as u64));
        assert_eq!(views_at(b, 3 * PAGE), [(b as usize, 3 * PAGE)]);
        // Another page in the middle: mapped, its neighbours kept.
        assert_eq!(XMemMapPhysicalPages(a.wrapping_byte_add(PAGE), 1, pages[2..].as_ptr()), a.wrapping_byte_add(PAGE));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { a.cast::<u8>().wrapping_add(PAGE).read() }, 7);
        assert_eq!(views_at(a, 3 * PAGE).len(), 3);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert!(map_alias(b as u64, 0, 3 * PAGE as u64));
        assert!(views_at(b, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading(first, 1);
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        // (Two sets of pages by turns, both starting with a 1: the same pages
        // where they are already are left alone, `VICTORY_REMAP_SAME`.)
        let mut others = [0usize; 3];
        let mut m = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut m, others.as_mut_ptr()), 1);
        assert!(write_physical(&others[..1], 0, &[1]));
        for i in 0..2000 {
            // The view goes and comes back, with a first byte of 1.
            let these = if i % 2 == 0 { &others } else { &pages };
            assert_eq!(XMemMapPhysicalPages(a, 3, these.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(3, others.as_ptr()), 1);
    }

    /// The views (address, bytes) over `[at, at + len)`.
    fn views_in(at: *mut c_void, len: usize) -> Vec<(usize, usize)> {
        let at = at as usize;
        physical().views.range(at..at + len).map(|(&base, &(size, _))| (base, size)).collect()
    }

    /// A pool of `n` pages mapped as one view (the title's linear pool view),
    /// each page's first byte its number, and a reservation of `n` 64 KiB
    /// placeholder pages (the GPU view).
    fn pool_and_view(n: usize) -> (Vec<usize>, *mut c_void, *mut c_void) {
        let pages = allocate(n);
        let reserve = || XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (pool, view) = (reserve(), reserve());
        assert!(!pool.is_null() && !view.is_null());
        assert_eq!(XMemMapPhysicalPages(pool, n, pages.as_ptr()), pool);
        for k in 0..n {
            // SAFETY: mapped read-write.
            unsafe { pool.cast::<u8>().add(k * PAGE).write(k as u8 + 1) };
        }
        (pages, pool, view)
    }

    /// The first byte of each page from `at` on, 0 for a page not mapped.
    fn first_bytes(at: *mut c_void, n: usize) -> Vec<u8> {
        (0..n)
            .map(|k| {
                let p = at.wrapping_byte_add(k * PAGE);
                // SAFETY: read only when committed.
                if query(p).host.State == MEM_COMMIT { unsafe { p.cast::<u8>().read() } } else { 0 }
            })
            .collect()
    }

    /// (F10, `VICTORY_MAP_DEFER`) Unmappings of both views deferred: the pages
    /// stay readable until applied; a map of the same page (or a copy from a
    /// pending source) applies them first, in call order; the end state is
    /// what the calls in order leave; DirectStorage writes are not told at a
    /// pending address.
    #[test]
    fn deferred_unmappings_end_as_in_order() {
        super::super::mapdefer::test_on(true);
        let (pages, pool, view) = pool_and_view(4);
        let (to, from) = (view as usize, pool as usize);
        assert!(map_alias(to as u64, from as u64, (4 * PAGE) as u64));
        assert_eq!(first_bytes(view, 4), [1, 2, 3, 4]);
        // GPU view: unmapped (deferred), then page 2 mapped again from pool page 1.
        assert!(map_alias(to as u64, 0, (4 * PAGE) as u64));
        assert!(map_alias((to + 2 * PAGE) as u64, from as u64, PAGE as u64));
        // CPU view (the pool): its page 3 unmapped by a scatter call.
        let address = [(from + 3 * PAGE) as *mut c_void];
        assert_eq!(XMemMapPhysicalPagesScatter(address.as_ptr(), 1, std::ptr::null()), 1);
        // A write to pool page 4 is not told at its pending CPU address.
        assert!(shown_at(&pages[3..4], 0, PAGE).iter().all(|r| r.0 != from + 3 * PAGE) || !DEFER.pending_at(from + 3 * PAGE));
        // Mapping it back where it was applies its unmapping first.
        assert_eq!(XMemMapPhysicalPagesScatter(address.as_ptr(), 1, pages[3..4].as_ptr()), 1);
        DEFER.drain();
        assert_eq!(first_bytes(view, 4), [0, 0, 1, 0]);
        assert_eq!(first_bytes(pool, 4), [1, 2, 3, 4]);
        // Unmapped for good, with nothing mapped after: gone once applied.
        assert!(map_alias((to + 2 * PAGE) as u64, 0, PAGE as u64));
        DEFER.drain();
        assert_eq!(first_bytes(view, 4), [0; 4]);
        super::super::mapdefer::test_on(false);
        assert_eq!(XMemMapPhysicalPages(pool, 4, std::ptr::null()), pool);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// (`VICTORY_MAP_PREFAULT`) A view mapped over resident pages, prefaulted,
    /// is read and written without a fault; a view unmapped again before the
    /// worker reaches it is skipped (nothing read, no crash; no lock held).
    #[test]
    fn prefaulted_views_take_no_first_touch_faults() {
        use std::sync::atomic::Ordering::Relaxed;
        let (pages, pool, view) = pool_and_view(4);
        // Resident through the lasting view (as DirectStorage writes them).
        assert!(write_physical(&pages, 0, &vec![5u8; 4 * PAGE]));
        let faults = victory_gpu::faults::process_faults;
        let full_pass = |at: usize| {
            let f = faults();
            for p in (at..at + 4 * PAGE).step_by(4096) {
                // SAFETY: a mapped read-write view.
                unsafe { std::ptr::read_volatile(p as *const u8) };
                unsafe { std::ptr::write_volatile((p + 1) as *mut u8, 3) };
            }
            faults() - f
        };
        // Without: a fresh view faults on each 4 KiB page.
        assert!(map_alias(view as u64, pool as u64, (4 * PAGE) as u64));
        let without = full_pass(view as usize);
        assert!(map_alias(view as u64, 0, (4 * PAGE) as u64));
        // With: queued at the map, read by the worker (here: this thread).
        super::super::prefault::TEST_ON.store(true, Relaxed);
        assert!(map_alias(view as u64, pool as u64, (4 * PAGE) as u64));
        super::super::prefault::drain_now();
        let with = full_pass(view as usize);
        // Unmapped before the worker reaches it: skipped.
        assert!(map_alias(view as u64, 0, (4 * PAGE) as u64));
        let line_before = super::super::prefault::take_line();
        assert_eq!(XMemMapPhysicalPages(view, 4, pages.as_ptr()), view);
        assert_eq!(XMemMapPhysicalPages(view, 4, std::ptr::null()), view);
        super::super::prefault::drain_now();
        let line = super::super::prefault::take_line();
        super::super::prefault::TEST_ON.store(false, Relaxed);
        println!("prefaulted views: faults of a full read+write pass without {without}, with {with}; {line_before} / {line}");
        assert!(without >= 60, "{without}");
        assert!(with < 8, "{with}");
        assert!(line_before.contains("prefaulted 64 pages in 1 ranges"), "{line_before}");
        assert!(line.contains("prefaulted 0 pages") && line.contains("skipped unmapped 64"), "{line}");
        assert_eq!(XMemMapPhysicalPages(pool, 4, std::ptr::null()), pool);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// (Prefault job) The adaptive discard delay.
    #[test]
    fn discard_delay_adapts_to_rewrites() {
        assert_eq!(next_delay(2000, 1000, 400, Some(8 << 30)), 4000);
        assert_eq!(next_delay(8000, 1000, 400, Some(8 << 30)), 10_000);
        assert_eq!(next_delay(8000, 1000, 10, Some(8 << 30)), 4000);
        assert_eq!(next_delay(2000, 1000, 10, Some(8 << 30)), 2000);
        assert_eq!(next_delay(4000, 1000, 100, Some(8 << 30)), 4000);
        assert_eq!(next_delay(8000, 10, 10, Some(8 << 30)), 8000, "too few to judge");
        assert_eq!(next_delay(8000, 1000, 900, Some(512 << 20)), 2000, "RAM short");
    }

    /// (F3, `VICTORY_MAP_FAST`) A run of copied pages over placeholders cut
    /// into pages by earlier page-by-page mappings is one view: the
    /// placeholders are joined first. Unmapped as one, it leaves one
    /// placeholder, and a page unmapped out of it keeps the rest.
    #[test]
    fn a_run_over_cut_placeholders_is_one_view() {
        let (pages, pool, view) = pool_and_view(8);
        let (to, from) = (view as usize, pool as usize);
        // Page by page first (the old way), then unmapped: eight placeholders.
        for k in 0..8 {
            assert!(map_alias_with(to + k * PAGE, from + k * PAGE, PAGE, false));
        }
        assert_eq!(views_in(view, 8 * PAGE).len(), 8);
        assert!(map_alias_with(to, 0, 8 * PAGE, false));
        assert_eq!(placeholder_pages(to + PAGE), 1, "cut into pages");
        // Pages 1..7 of the view from pool pages 2..8, as one run.
        assert!(map_alias_with(to + PAGE, from + 2 * PAGE, 6 * PAGE, true));
        assert_eq!(views_in(view, 8 * PAGE), [(to + PAGE, 6 * PAGE)]);
        assert_eq!(first_bytes(view, 8), [0, 3, 4, 5, 6, 7, 8, 0]);
        // The pages say where they show, as page by page.
        assert_eq!(shown_at(&pages[2..3], 0, PAGE).iter().filter(|r| r.0 == to + PAGE).count(), 1);
        // One page out of the middle: the rest stays (cut, mapped again).
        assert!(map_alias_with(to + 3 * PAGE, 0, PAGE, true));
        assert_eq!(first_bytes(view, 8), [0, 3, 4, 0, 6, 7, 8, 0]);
        assert_eq!(views_in(view, 8 * PAGE), [(to + PAGE, 2 * PAGE), (to + 4 * PAGE, 3 * PAGE)]);
        // All of it unmapped: one call each, nothing left.
        assert!(map_alias_with(to, 0, 8 * PAGE, true));
        assert!(views_in(view, 8 * PAGE).is_empty());
        assert_eq!(first_bytes(view, 8), [0; 8]);
        // Mapped as one run across what were three placeholders, at the start
        // of the reservation: still one view.
        assert!(map_alias_with(to, from, 8 * PAGE, true));
        assert_eq!(views_in(view, 8 * PAGE), [(to, 8 * PAGE)]);
        assert_eq!(first_bytes(view, 8), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(map_alias_with(to, 0, 8 * PAGE, true));
        assert_eq!(XMemMapPhysicalPages(pool, 8, std::ptr::null()), pool);
        assert_eq!(XMemFreePhysicalPages(8, pages.as_ptr()), 1);
    }

    /// (F3, `VICTORY_MAP_FAST`) The same steps as runs and page by page leave
    /// the same bytes at the same addresses, and the same pages shown where.
    #[test]
    fn runs_and_pages_end_the_same() {
        let steps: [(usize, usize, usize); 7] = [
            // (destination page, pool page + 1 or 0 to unmap, pages)
            (0, 1, 4),
            (4, 7, 2),
            (2, 0, 3),
            (1, 5, 3),
            (3, 7, 2), // its first page there already
            (6, 3, 2),
            (0, 0, 2),
        ];
        let mut ends = Vec::new();
        for fast in [false, true] {
            let (pages, pool, view) = pool_and_view(8);
            let (to, from) = (view as usize, pool as usize);
            for &(d, p, n) in &steps {
                let source = if p == 0 { 0 } else { from + (p - 1) * PAGE };
                if fast {
                    assert!(map_alias_with(to + d * PAGE, source, n * PAGE, true));
                } else {
                    for k in 0..n {
                        assert!(map_alias_with(to + (d + k) * PAGE, if p == 0 { 0 } else { source + k * PAGE }, PAGE, false));
                    }
                }
            }
            let shown: Vec<Vec<(usize, usize)>> =
                (0..8).map(|k| shown_at(&pages[k..k + 1], 0, PAGE).into_iter().filter(|r| r.0 >= to && r.0 < to + 8 * PAGE).map(|(a, b)| (a - to, b)).collect()).collect();
            ends.push((first_bytes(view, 8), shown));
            assert!(map_alias_with(to, 0, 8 * PAGE, fast));
            assert_eq!(XMemMapPhysicalPages(pool, 8, std::ptr::null()), pool);
            assert_eq!(XMemFreePhysicalPages(8, pages.as_ptr()), 1);
        }
        assert_eq!(ends[0], ends[1]);
        assert_eq!(ends[0].0, [0, 0, 6, 7, 8, 8, 3, 4]);
    }

    /// (F3, `VICTORY_MAP_FAST`) Scatter calls as runs and page by page leave
    /// the same bytes at the same addresses and the same pages shown where;
    /// a run of pages that follow one another in their section is one view.
    #[test]
    fn scatter_runs_and_pages_end_the_same() {
        // (destination page, pool page + 1 or 0 to unmap) per entry, per call.
        let calls: [&[(usize, usize)]; 6] = [
            &[(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)],
            &[(1, 0), (2, 0)],
            &[(1, 7), (2, 8), (3, 4), (4, 5)], // page 3 there already
            &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
            &[(2, 3), (3, 4), (4, 5), (5, 6)],
            &[(3, 0)],
        ];
        let mut ends = Vec::new();
        for fast in [false, true] {
            let pages = allocate(8);
            let view = XMemVirtualAlloc(std::ptr::null_mut(), 8 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!view.is_null());
            for k in 0..8 {
                assert!(write_physical(&pages[k..k + 1], 0, &[k as u8 + 1]));
            }
            let mut views = Vec::new();
            for call in calls {
                let addresses: Vec<*mut c_void> = call.iter().map(|&(d, _)| view.wrapping_byte_add(d * PAGE)).collect();
                let numbers: Vec<usize> = call.iter().map(|&(_, p)| if p == 0 { 0 } else { pages[p - 1] }).collect();
                let ok = if fast {
                    scatter_runs(addresses.as_ptr(), call.len(), numbers.as_ptr()).0
                } else {
                    scatter_pages(addresses.as_ptr(), call.len(), numbers.as_ptr())
                };
                assert!(ok);
                views.push(views_in(view, 8 * PAGE).len());
            }
            let to = view as usize;
            let shown: Vec<Vec<(usize, usize)>> = (0..8)
                .map(|k| shown_at(&pages[k..k + 1], 0, PAGE).into_iter().filter(|r| r.0 >= to && r.0 < to + 8 * PAGE).map(|(a, b)| (a - to, b)).collect())
                .collect();
            ends.push((first_bytes(view, 8), shown));
            if fast {
                // The first call: pages 1-4 as one view, page 6 alone.
                assert_eq!(views[0], 2);
            }
            assert_eq!(XMemMapPhysicalPages(view, 8, std::ptr::null()), view);
            assert_eq!(XMemFreePhysicalPages(8, pages.as_ptr()), 1);
        }
        assert_eq!(ends[0], ends[1]);
        assert_eq!(ends[0].0, [0, 0, 3, 0, 5, 6, 0, 0]);
    }

    #[test]
    fn views_that_follow_on_are_one_run() {
        assert_eq!(view_runs([(0, 2), (2, 1), (5, 1), (6, 4)].into_iter()), [(0, 3), (5, 10)]);
        assert!(view_runs(std::iter::empty()).is_empty());
    }

    /// (F3) The floor of the host calls a page costs: map (cut, view,
    /// commit) and unmap, page by page and as one run of 64 pages. Prints;
    /// `cargo test -p victory-hle --lib host_call_floor -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn host_call_floor() {
        const N: usize = 256;
        let (pages, pool, view) = pool_and_view(N);
        let (to, from) = (view as usize, pool as usize);
        for (label, run) in [("page by page", 1), ("runs of 64", 64)] {
            let t = std::time::Instant::now();
            for round in 0..8 {
                for k in (0..N).step_by(run) {
                    assert!(map_alias_with(to + k * PAGE, from + ((k + round) % (N - run + 1)) * PAGE, run * PAGE, true));
                }
                for k in (0..N).step_by(run) {
                    assert!(map_alias_with(to + k * PAGE, 0, run * PAGE, true));
                }
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / (8 * N) as f64;
            println!("{label}: {us:.1} us per page mapped and unmapped = {:.0} ms per 1,000 pages", us);
        }
        assert_eq!(XMemMapPhysicalPages(pool, N, std::ptr::null()), pool);
        assert_eq!(XMemFreePhysicalPages(N, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }
}

/// The space below 4 GiB, which the title's 32-bit memory (`XMemVirtualAlloc`
/// with the 32-bit address flag) has to fit in.
const FOUR_GIB: usize = 1 << 32;

/// Guards over the space below 4 GiB. Windows places allocations bottom up,
/// so the host's own memory (the driver's, heaps, the shader and pipeline
/// caches read at startup) filled it first: with a large pipeline cache only
/// 22 MB was left, in pieces, when Grand Theft Auto VI asked for a 185 MB
/// 32-bit placeholder at startup, the request was refused
/// (`ERROR_NOT_ENOUGH_MEMORY`) and the title stopped at an assertion. The
/// debug layer's memory did the same. So the free space below 4 GiB is
/// reserved at startup, past [`low_guards::SLACK`] left for the host, and a
/// 32-bit request that finds no room releases guards and tries again.
pub mod low_guards {
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    use super::FOUR_GIB;

    /// Free space left to the host below 4 GiB (the lowest free space).
    pub(crate) const SLACK: usize = 256 << 20;
    /// The allocation granularity of Windows.
    const GRANULE: usize = 64 << 10;

    /// The guards, `(base, bytes)`.
    static GUARDS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    fn guards() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
        GUARDS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The free runs in `[from, to)`, granule-aligned.
    fn free_runs(from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut at = from.max(GRANULE);
        while at < to {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            let base = m.BaseAddress as usize;
            let end = (base + m.RegionSize).min(to);
            if m.State == MEM_FREE {
                let (start, end) = (
                    base.max(from).next_multiple_of(GRANULE),
                    end & !(GRANULE - 1),
                );
                if end > start {
                    runs.push((start, end - start));
                }
            }
            at = base + m.RegionSize;
        }
        runs
    }

    /// Reserves `[base, base + bytes)` as a guard; false if something took it.
    fn guard(base: usize, bytes: usize) -> bool {
        // SAFETY: a reservation of free address space; nothing refers to it.
        let at = unsafe { VirtualAlloc(base as *const _, bytes, MEM_RESERVE, PAGE_NOACCESS) };
        at as usize == base
    }

    /// Reserves the free space below 4 GiB but the lowest [`SLACK`]. Call
    /// once, before the host allocates much (the GPU device, the caches).
    pub fn install() {
        let mut guards = guards();
        let mut slack = SLACK;
        for (mut base, mut bytes) in free_runs(0, FOUR_GIB) {
            let left = slack.min(bytes);
            (base, bytes, slack) = (base + left, bytes - left, slack - left);
            if bytes > 0 && guard(base, bytes) {
                guards.push((base, bytes));
            }
        }
        let total: usize = guards.iter().map(|g| g.1).sum();
        tracing::info!(
            guards = guards.len(),
            mb = total >> 20,
            "space below 4 GiB kept for the title"
        );
    }

    /// Try the smallest fitting guards first, then try smaller guards in
    /// descending size order. A replacement guard cannot be retried here.
    pub(crate) fn retry(
        bytes: usize,
        mut allocate: impl FnMut() -> *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};

        // Preserve the allocation error across address queries and reservations.
        // SAFETY: reads this thread's last-error value.
        let mut error = unsafe { GetLastError() };
        let mut guards = guards();
        let mut candidates = guards.clone();
        candidates.sort_unstable_by_key(|&(_, size)| {
            let fits = size >= bytes;
            (!fits, if fits { size } else { usize::MAX - size })
        });
        for (base, size) in candidates {
            // SAFETY: our own reservation, with no allocations inside it.
            if unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } == 0 {
                continue;
            }
            guards.retain(|&(b, _)| b != base);
            let at = allocate();
            // SAFETY: reads this thread's last-error value.
            error = unsafe { GetLastError() };
            for (start, len) in free_runs(base, base + size) {
                if guard(start, len) {
                    guards.push((start, len));
                }
            }
            if !at.is_null() {
                return at;
            }
        }
        // SAFETY: sets this thread's last-error value.
        unsafe { SetLastError(error) };
        std::ptr::null_mut()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        static TEST: Mutex<()> = Mutex::new(());

        #[test]
        fn a_refused_retry_leaves_the_guards_in_place() {
            let _one = TEST.lock().unwrap();
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) };
            assert!(!hole.is_null());
            let first = hole as usize;
            // SAFETY: ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
            assert!(guard(first, 4 * GRANULE));
            guards().push((first, 4 * GRANULE));
            let mut attempts = 0;
            let result = retry(4 * GRANULE, || {
                attempts += 1;
                std::ptr::null_mut()
            });
            assert!(result.is_null());
            assert_eq!(attempts, 1);
            assert!(guards().contains(&(first, 4 * GRANULE)));
            guards().retain(|&(base, _)| base != first);
            // SAFETY: the guard the retry put back, ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
        }

        /// A released guard is guarded again around what took part of it.
        #[test]
        fn a_released_guard_is_guarded_again_around_what_took_it() {
            let _one = TEST.lock().unwrap();
            // A known hole of four granules: reserved, then freed.
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(hole, 0);
            // SAFETY: ours.
            unsafe { VirtualFree(hole as *mut _, 0, MEM_RELEASE) };
            assert!(guard(hole, 4 * GRANULE));
            guards().push((hole, 4 * GRANULE));
            let taken = retry(GRANULE, || {
                // SAFETY: a reservation of the first granule at a known address.
                unsafe { VirtualAlloc(hole as *const _, GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
            }) as usize;
            assert_eq!(taken, hole);
            assert!(guards().contains(&(hole + GRANULE, 3 * GRANULE)));
            guards().retain(|&(b, _)| b != hole + GRANULE);
            // SAFETY: ours.
            unsafe {
                VirtualFree(taken as *mut _, 0, MEM_RELEASE);
                VirtualFree((hole + GRANULE) as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}
