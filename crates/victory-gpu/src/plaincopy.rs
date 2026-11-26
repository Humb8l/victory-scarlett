//! Copies of title memory made in place, by this process, with a way back
//! when a page is not there (`VICTORY_COPY_PLAIN=1`; off by default).
//!
//! Every copy of title memory into a host buffer goes through the kernel
//! (`ReadProcessMemory` on our own process, `crate::host::copy_mapped`):
//! the title unmaps and remaps its memory while we read it, a read in
//! place of a page that is gone is an access violation, and the kernel's
//! copy fails instead. It costs what a system call that locks and probes
//! every page costs: run 225's profile has the submitting thread 35 ms a
//! frame inside `ZwReadVirtualMemory` for about 100 MB (3 GB a second),
//! where the processor's own copy moves that much in a third of the time.
//!
//! Here the copy is one `rep movsb` in a function of ours that uses no
//! stack. An exception handler, first in the line, looks at every access
//! violation: one that is a READ and happened at exactly that instruction
//! continues at the function's other exit, which returns 0, and the
//! caller falls back to the kernel's copy as before (it says which page
//! is gone and fills in zeros). A write that faults is not ours to take:
//! the destination is host memory, which is always there; such a fault
//! goes on to the handlers after this one.
//!
//! Writes (`VICTORY_WRITE_PLAIN`, after ShakiraRED-dt's plaincopy.rs): the
//! copy-back's writes into title memory (`crate::host::write_mapped`,
//! some 60 MB a frame) the same way, with a second such function whose
//! WRITE faults (a page that is not there or not writable) end it with 0;
//! the caller then writes the whole range through the kernel as before,
//! which rewrites what the copy in place wrote before the fault with the
//! same bytes and leaves out what it cannot. Write tracking sees these
//! writes as it saw the kernel's: the callers open the guards of the range
//! first (`crate::watch::writing`), and the write watch notes a write in
//! place as it noted the kernel's.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_copy_guarded",
    ".globl victory_copy_guarded_at",
    ".globl victory_copy_guarded_fault",
    "victory_copy_guarded:",
    // rcx: to, rdx: from, r8: bytes. rsi and rdi are the caller's: kept in
    // r9 and r10, which are ours to use, so nothing is on the stack.
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_copy_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_copy_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
    // The same for writes into title memory, at an instruction of its own.
    ".globl victory_write_guarded",
    ".globl victory_write_guarded_at",
    ".globl victory_write_guarded_fault",
    "victory_write_guarded:",
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_write_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_write_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_copy_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_guarded_at: u8;
    static victory_copy_guarded_fault: u8;

    fn victory_write_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_write_guarded_at: u8;
    static victory_write_guarded_fault: u8;
}

/// Whether copies of title memory into host memory are made in place
/// (`VICTORY_COPY_PLAIN=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(target_arch = "x86_64") && std::env::var_os("VICTORY_COPY_PLAIN").is_some_and(|v| v == "1");
        if on {
            install();
            tracing::info!(
                "copies of title memory into host buffers are made in place, with the kernel's copy behind them for a \
                 page that is not there (VICTORY_COPY_PLAIN=1)"
            );
        }
        on
    })
}

/// Whether the copy-back's writes into title memory are made in place
/// ([`write`]): `VICTORY_WRITE_PLAIN=1` on, `=0` off, unset as
/// `VICTORY_COPY_PLAIN` ([`on`]).
pub(crate) fn write_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(target_arch = "x86_64")
            && match std::env::var_os("VICTORY_WRITE_PLAIN") {
                Some(v) => v == "1",
                None => on(),
            };
        if on {
            install();
            tracing::info!(
                "copy-back writes into title memory are made in place, with the kernel's write behind them for a page \
                 that is not there or not writable (VICTORY_WRITE_PLAIN)"
            );
        }
        on
    })
}

/// Writes made in place, their bytes, the writes that met a page that was
/// not there or not writable (and went to the kernel's write), and the
/// writes not tried in place because their destination faulted lately
/// ([`WRECENT`]), since the last line.
static WCOUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// As [`RECENT`], for the destinations of writes in place.
static WRECENT: [std::sync::atomic::AtomicUsize; 1024] = [const { std::sync::atomic::AtomicUsize::new(0) }; 1024];
/// When [`WRECENT`] was last emptied.
static WEMPTIED: AtomicU64 = AtomicU64::new(0);

/// Copies made in place, their bytes, the copies that met a page that was
/// not there (and went to the kernel's copy), and the copies not tried in
/// place because their source faulted lately ([`RECENT`]) or they were
/// under [`LEAST`] bytes, since the last line.
static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The fewest bytes copied in place. A read of a byte or a word is how the
/// layer ASKS whether a page is there (a texture's first page, a record):
/// it is expected to fail, and a failed kernel read costs a microsecond
/// where a fault and its handler cost tens (run 230: 1,000 to 1,300 faults
/// a frame, nearly all of them such askings, took back most of what the
/// copies in place had won).
const LEAST: usize = 256;

/// The 64 KiB blocks of title memory a copy in place faulted in lately:
/// their copies go straight to the kernel's, which reads round the hole.
/// By the block's number, in the place its low bits name (a later block
/// with the same low bits takes the place). Forgotten every two seconds:
/// the title maps the memory in the end.
static RECENT: [std::sync::atomic::AtomicUsize; 1024] = [const { std::sync::atomic::AtomicUsize::new(0) }; 1024];
/// When [`RECENT`] was last emptied (milliseconds of [`now_ms`]).
static EMPTIED: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// The numbers of the first and last 64 KiB block of `[from, from + len)`,
/// never 0 (0 is an empty place).
fn blocks(from: usize, len: usize) -> (usize, usize) {
    ((from >> 16) | 1 << 63, ((from + len.max(1) - 1) >> 16) | 1 << 63)
}

/// Whether a copy from there faulted lately.
fn faulted_lately(from: usize, len: usize) -> bool {
    faulted_lately_in(&RECENT, &EMPTIED, from, len)
}

/// [`faulted_lately`] in `table`, emptied at `emptied_at` ([`RECENT`] or
/// [`WRECENT`]).
fn faulted_lately_in(
    table: &[std::sync::atomic::AtomicUsize; 1024],
    emptied_at: &AtomicU64,
    from: usize,
    len: usize,
) -> bool {
    let now = now_ms();
    let emptied = emptied_at.load(Relaxed);
    if now.saturating_sub(emptied) >= 2_000 && emptied_at.compare_exchange(emptied, now, Relaxed, Relaxed).is_ok() {
        for place in table {
            place.store(0, Relaxed);
        }
        return false;
    }
    let (first, last) = blocks(from, len);
    table[first & 1023].load(Relaxed) == first || table[last & 1023].load(Relaxed) == last
}

/// A copy from there faulted.
fn note_fault(from: usize, len: usize) {
    note_fault_in(&RECENT, from, len);
}

/// [`note_fault`] in `table`.
fn note_fault_in(table: &[std::sync::atomic::AtomicUsize; 1024], from: usize, len: usize) {
    let (first, last) = blocks(from, len);
    table[first & 1023].store(first, Relaxed);
    table[last & 1023].store(last, Relaxed);
}

/// The handler that turns a read fault inside [`guarded`] into its return
/// of false; put first in the line, once.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it will continue with.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        // A write in place ([`guarded_write`]): a WRITE fault at its own
        // instruction ends it, and the caller writes through the kernel.
        // (Put first in the line when installed: if the guards' handler
        // came later and stands before this one, a block it guards is
        // opened by it as `crate::watch::writing` opened the range just
        // before, and the write goes on.)
        if context.Rip == (&raw const victory_write_guarded_at) as u64 {
            const WRITE: usize = 1;
            let write = r.NumberParameters >= 1 && r.ExceptionInformation[0] == WRITE;
            let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && write;
            if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
                context.Rip = (&raw const victory_write_guarded_fault) as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // Only at the copy's own instruction.
        if context.Rip != (&raw const victory_copy_guarded_at) as u64 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const READ: usize = 0;
        let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
        let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
        if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
            context.Rip = (&raw const victory_copy_guarded_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that reads the record it is given and nothing else.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn install() {}

/// Copies `len` bytes from `from` to `to` in place; false if a page of the
/// source was not there (then some of `to` may have been written).
///
/// `to` must be writable memory of `len` bytes that is ours (host memory);
/// `from` may be anything.
#[cfg(target_arch = "x86_64")]
fn guarded(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the destination is the caller's to write; a source page that
    // is not there ends the copy with 0 instead of a fault ([`install`]).
    unsafe { victory_copy_guarded(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// (`VICTORY_MAP_PREFAULT`) Reads one byte of each 4 KiB page of
/// `[at, at + len)`, in order, with no lock: a page that is not there (a
/// view taken away meanwhile, a placeholder, no access) ends it without a
/// fault reaching anyone. The pages read, and whether it ended early.
pub(crate) fn touch(at: usize, len: usize) -> (usize, bool) {
    let mut b = 0u8;
    let mut read = 0;
    let mut p = at & !4095;
    while p < at.saturating_add(len) {
        if !guarded(&mut b, p.max(at) as *const u8, 1) {
            return (read, true);
        }
        read += 1;
        p += 4096;
    }
    (read, false)
}

/// Copies `len` bytes from `from` (host memory of ours) to `to` (title
/// memory) in place; false if a page of the destination was not there or
/// not writable (then some of `to` may have been written).
#[cfg(target_arch = "x86_64")]
fn guarded_write(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the source is the caller's to read; a destination page that
    // is not there or not writable ends the copy with 0 instead of a fault
    // ([`install`]).
    unsafe { victory_write_guarded(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded_write(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// With [`write_on`]: copies host memory at `from` into title memory at
/// `to`; false if it is off, the destination faulted lately, or a page of
/// it was not there or not writable (the caller then writes the whole
/// range through the kernel, as before).
pub(crate) fn write(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !write_on() || len == 0 {
        return false;
    }
    if faulted_lately_in(&WRECENT, &WEMPTIED, to as usize, len) {
        WCOUNT[3].fetch_add(1, Relaxed);
        return false;
    }
    let whole = guarded_write(to, from, len);
    if whole {
        WCOUNT[0].fetch_add(1, Relaxed);
        WCOUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        WCOUNT[2].fetch_add(1, Relaxed);
        note_fault_in(&WRECENT, to as usize, len);
    }
    whole
}

/// With [`on`]: copies title memory at `from` into host memory at `to`;
/// false if it is off or a source page was not there (the caller then
/// copies through the kernel, as before).
pub(crate) fn copy(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    // (F3, `VICTORY_LAZY_GPU_VIEW`) A read of the GPU view, for its meter.
    crate::gpuview::note_read(from as usize, len);
    if len < LEAST {
        COUNT[4].fetch_add(1, Relaxed);
        return false;
    }
    if faulted_lately(from as usize, len) {
        COUNT[3].fetch_add(1, Relaxed);
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT[0].fetch_add(1, Relaxed);
        COUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        COUNT[2].fetch_add(1, Relaxed);
        note_fault(from as usize, len);
    }
    whole
}

/// With the frame line: what [`copy`] did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if write_on() {
        let [writes, bytes, gone, lately] = [0, 1, 2, 3].map(|i| WCOUNT[i].swap(0, Relaxed) as f64 / frames);
        tracing::info!(
            "copy-back writes into title memory made in place per frame (VICTORY_WRITE_PLAIN): {writes:.0}, {:.1} MB; \
             {gone:.1} met a page that was not there or not writable and went through the kernel's write; left to the \
             kernel's write untried: {lately:.0} whose destination faulted in the last two seconds",
            bytes / 1e6
        );
    }
    if !on() {
        return;
    }
    let [copies, bytes, gone, lately, small] = [0, 1, 2, 3, 4].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "copies of title memory made in place per frame (VICTORY_COPY_PLAIN=1): {copies:.0}, {:.1} MB; {gone:.1} met a page \
         that was not there and went through the kernel's copy; left to the kernel's copy untried: {lately:.0} whose source \
         faulted in the last two seconds, {small:.0} of under {LEAST} bytes",
        bytes / 1e6
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    #[test]
    fn memory_that_is_there_is_copied() {
        let from: Vec<u8> = (0..100_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut to = vec![0u8; from.len()];
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), from.len()));
        assert_eq!(to, from);
        // Nothing to copy is a copy.
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), 0));
    }

    /// Three pages reserved, the first committed: a copy that stays in the
    /// first page is whole; one that runs into the second ends with false,
    /// the bytes before the hole copied, and the thread goes on.
    #[test]
    fn a_page_that_is_not_there_ends_the_copy_without_a_fault() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
        let mut to = vec![0u8; 2 * PAGE];
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        to.fill(0);
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        assert!(to[..PAGE / 2].iter().all(|&b| b == 0xAB), "the bytes before the hole");
        assert!(to[PAGE / 2..].iter().all(|&b| b == 0), "nothing after it");
        // A source that is not there at all.
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));
        // Committed but closed to reads.
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(2 * PAGE), 16));
        // And the registers the caller owns came back: this still works.
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// Writes in place: whole into committed read-write memory; a page that
    /// is not there or is read-only ends the write without a fault, the
    /// bytes before it written (ShakiraRED-dt's test, extended).
    #[test]
    fn guarded_writes_end_at_a_page_that_is_not_there_or_not_writable() {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first and the third page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READONLY) }.is_null());
        let src = vec![0x5Au8; 2 * PAGE];
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        let mut dst = vec![0u8; PAGE];
        assert!(guarded(dst.as_mut_ptr(), base, PAGE));
        assert_eq!(dst, src[..PAGE]);
        // Into the hole from the middle of the first page: the half before
        // it is written, and the thread goes on.
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0);
        assert!(!guarded_write(base.wrapping_add(PAGE / 2), src.as_ptr(), PAGE));
        // SAFETY: committed and readable.
        let first = unsafe { std::slice::from_raw_parts(base, PAGE) };
        assert!(first[..PAGE / 2].iter().all(|&b| b == 0) && first[PAGE / 2..].iter().all(|&b| b == 0x5A));
        // Not there at all; read-only (readable, not writable).
        assert!(!guarded_write(base.wrapping_add(PAGE), src.as_ptr(), PAGE));
        assert!(guarded(dst.as_mut_ptr(), base.wrapping_add(2 * PAGE), PAGE));
        assert!(!guarded_write(base.wrapping_add(2 * PAGE), src.as_ptr(), 16));
        // The registers came back: this still works.
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// A write in place that faults midway is completed the old way:
    /// `crate::host::write_mapped` then writes the whole range through the
    /// kernel, which writes every page that is there (after the hole too)
    /// and leaves the hole alone.
    #[test]
    fn a_write_that_faults_midway_is_completed_through_the_kernel() {
        // SAFETY: memory of the test's own: page 0 and 2 committed, 1 not.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let src: Vec<u8> = (0..3 * PAGE as u32).map(|i| (i * 13 + 1) as u8).collect();
        assert!(!guarded_write(base, src.as_ptr(), 3 * PAGE), "the hole ends the write in place");
        // SAFETY: committed pages of the reservation.
        let (first, third) = unsafe {
            (std::slice::from_raw_parts(base, PAGE), std::slice::from_raw_parts(base.wrapping_add(2 * PAGE), PAGE))
        };
        assert_eq!(first, &src[..PAGE]);
        assert!(third.iter().all(|&b| b == 0), "not reached in place");
        crate::host::write_mapped(base, src.as_ptr(), 3 * PAGE);
        assert_eq!(first, &src[..PAGE]);
        assert_eq!(third, &src[2 * PAGE..], "the kernel's write went on past the hole");
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// A source that faulted is remembered by its 64 KiB blocks, and only
    /// those.
    #[test]
    fn a_source_that_faulted_is_remembered_by_its_blocks() {
        // (Addresses no other test uses; the table is shared.)
        let at = 0x7654_3210_0000usize;
        assert!(!faulted_lately(at, 4096));
        note_fault(at + 100, 4096);
        assert!(faulted_lately(at, 16));
        assert!(faulted_lately(at + 0xffff, 1));
        // A copy that ends in the block, and one that begins in it.
        assert!(faulted_lately(at - 4096, 8192));
        // The blocks beside it are not remembered.
        assert!(!faulted_lately(at + 0x10000, 4096));
        assert!(!faulted_lately(at - 0x10000, 4096));
        assert_eq!(blocks(0, 0), (1 << 63, 1 << 63));
    }

    /// Many threads at once, half of them faulting: each gets its own answer.
    #[test]
    fn threads_fault_and_copy_side_by_side() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(base, 0);
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut to = vec![0u8; PAGE];
                    for _ in 0..2_000 {
                        let from = (base + if t % 2 == 0 { 0 } else { PAGE }) as *const u8;
                        assert_eq!(guarded(to.as_mut_ptr(), from, PAGE), t % 2 == 0);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}
