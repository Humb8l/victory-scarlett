//! (Card kinds job, meter 5 of `docs/frame-pipeline-analysis.md`) The
//! process's processor time by thread group, for the frame log: the title's
//! recording threads (those seen making list calls, diagnostics builds), the
//! title's other threads, our submit thread, our relay / copy-back threads,
//! storage (DirectStorage workers, mapping and prefault threads), and our
//! other threads. Every thread of the process is read about once a second
//! (a snapshot of the system's threads, one handle each kept open), and the
//! differences go to the frame log's totals, user and kernel time together.

use std::sync::Mutex;

use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next};
use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetThreadDescription, GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};

/// The groups, in the frame log's order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Group {
    Recording,
    Title,
    Submit,
    Relay,
    Storage,
    Ours,
}

/// The group of a thread by its name (`recording`: seen making list calls).
pub(crate) fn group(name: &str, recording: bool) -> Group {
    if recording {
        return Group::Recording;
    }
    let starts = |p: &[&str]| p.iter().any(|p| name.starts_with(p));
    if name == "victory-submit" {
        Group::Submit
    } else if starts(&["victory-relay", "victory-late-wait", "victory-wide", "victory-card-clock", "victory-queue2"]) {
        Group::Relay
    } else if starts(&["dstorage", "victory-map", "victory-prefault", "victory-pool-discard", "victory-memory-map"]) {
        Group::Storage
    } else if starts(&["victory", "card accounting"]) {
        Group::Ours
    } else {
        Group::Title
    }
}

struct Seen {
    handle: HANDLE,
    name: String,
    /// User + kernel microseconds when last read.
    last: u64,
    /// Read in the last sweep.
    alive: bool,
}

// SAFETY: thread handles are plain kernel handles, used behind a lock.
unsafe impl Send for Seen {}

struct State {
    at: std::time::Instant,
    threads: std::collections::HashMap<u32, Seen>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn micros(t: FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

fn name_of(handle: HANDLE) -> String {
    let mut text: windows_sys::core::PWSTR = std::ptr::null_mut();
    // SAFETY: a thread handle with query rights; the text is freed below.
    let hr = unsafe { GetThreadDescription(handle, &mut text) };
    if hr < 0 || text.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated wide string the call allocated.
    let name = unsafe {
        let mut n = 0;
        while *text.add(n) != 0 {
            n += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(text, n))
    };
    // SAFETY: memory the call allocated with LocalAlloc.
    unsafe { LocalFree(text.cast()) };
    name
}

fn times(handle: HANDLE) -> Option<u64> {
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: a thread handle with query rights and four out pointers.
    let ok = unsafe { GetThreadTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) };
    (ok != 0).then(|| micros(user) + micros(kernel))
}

/// Called at every frame's end; reads the threads about once a second.
pub(crate) fn sample() {
    use crate::framelog::{T, add};
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = state.get_or_insert_with(|| State {
        at: std::time::Instant::now() - std::time::Duration::from_secs(2),
        threads: Default::default(),
    });
    if s.at.elapsed() < std::time::Duration::from_secs(1) {
        return;
    }
    s.at = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    let recorders: std::collections::HashSet<u32> = crate::copystat::recorder_ids().into_iter().collect();
    #[cfg(not(feature = "diagnostics"))]
    let recorders: std::collections::HashSet<u32> = Default::default();
    // SAFETY: plain queries.
    let (pid, snap) = unsafe { (GetCurrentProcessId(), CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)) };
    if snap == INVALID_HANDLE_VALUE || snap.is_null() {
        return;
    }
    for seen in s.threads.values_mut() {
        seen.alive = false;
    }
    let mut sums = [0u64; 6];
    // SAFETY: plain data the calls fill in.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: our snapshot and a sized entry.
    let mut more = unsafe { Thread32First(snap, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == pid {
            let id = entry.th32ThreadID;
            let seen = match s.threads.entry(id) {
                std::collections::hash_map::Entry::Occupied(e) => Some(e.into_mut()),
                std::collections::hash_map::Entry::Vacant(e) => {
                    // SAFETY: a plain open of a thread of ours.
                    let handle = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, id) };
                    if handle.is_null() {
                        None
                    } else {
                        // A thread first seen: its time so far is not this second's.
                        let last = times(handle).unwrap_or(0);
                        Some(e.insert(Seen { handle, name: name_of(handle), last, alive: true }))
                    }
                }
            };
            if let Some(seen) = seen {
                seen.alive = true;
                if let Some(now) = times(seen.handle) {
                    // Title threads may be named late.
                    if !seen.name.starts_with("victory") && !seen.name.starts_with("dstorage") {
                        seen.name = name_of(seen.handle);
                    }
                    let g = group(&seen.name, recorders.contains(&id));
                    sums[g as usize] += now.saturating_sub(seen.last);
                    seen.last = now;
                }
            }
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snap, &mut entry) } != 0;
    }
    // SAFETY: our snapshot.
    unsafe { CloseHandle(snap) };
    s.threads.retain(|_, seen| {
        if !seen.alive {
            // SAFETY: a handle we opened.
            unsafe { CloseHandle(seen.handle) };
        }
        seen.alive
    });
    for (t, n) in [T::CpuRecordUs, T::CpuTitleUs, T::CpuSubmitUs, T::CpuRelayUs, T::CpuStorageUs, T::CpuOursUs].into_iter().zip(sums) {
        add(t, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_go_to_their_groups_by_name() {
        assert_eq!(group("anything", true), Group::Recording);
        assert_eq!(group("victory-submit", false), Group::Submit);
        assert_eq!(group("victory-relay", false), Group::Relay);
        assert_eq!(group("victory-wide-1-3", false), Group::Relay);
        assert_eq!(group("dstorage 4", false), Group::Storage);
        assert_eq!(group("victory-prefault", false), Group::Storage);
        assert_eq!(group("victory-pipelines-3", false), Group::Ours);
        assert_eq!(group("Subrender:03", false), Group::Title);
        assert_eq!(group("", false), Group::Title);
    }

    #[test]
    fn a_sample_reads_this_process_threads() {
        sample();
        let s = STATE.lock().unwrap();
        assert!(!s.as_ref().unwrap().threads.is_empty());
    }
}
