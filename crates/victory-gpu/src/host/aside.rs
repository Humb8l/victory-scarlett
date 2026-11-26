//! (P6, port of avatarmanz f672130) Work for a list that a thread of ours
//! does while the title records on: the pass over the heap's buffer views
//! (`crate::draw`, `State::heap_buffers`), which took 12 ms a frame at the
//! median, 25 ms at the 90th (run-90, "heap buffers ... passes ... ms") of
//! one recording thread at the frame's first list that needs it.
//! `VICTORY_HEAP_PASS_ASIDE=0`: done where it is needed, as before.
//!
//! What it finds are refreshes of host copies, which are steps of the list
//! for its submission, and what it makes are host descriptors, which the
//! card reads when the list runs: both are there when the list is closed
//! ([`super::HostList::close`] waits), and the refreshes stand at the place
//! among the list's steps where the pass was begun, deduplicated as
//! `HostList::defer_refresh` would have done there
//! ([`super::HostList::take_coming`]). What the pass leaves for the draws
//! that follow (the heap's unordered views, the spans, the marks) is read
//! only after it is done ([`wait_pass`]).

use super::*;

/// A host copy, the title address, the bytes (as `Shadow::refresh` cuts them).
pub(crate) type Refresh = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// What is being found, until it is: done, and what was found (taken once).
pub(crate) struct Coming(Mutex<(bool, Option<Vec<Refresh>>)>, std::sync::Condvar);

/// Since the last frame line: microseconds lists (or draws reading what a
/// pass leaves) waited for an aside pass; passes made aside.
pub(crate) static WAITED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Coming {
    fn done(&self) -> std::sync::MutexGuard<'_, (bool, Option<Vec<Refresh>>)> {
        let began = std::time::Instant::now();
        let mut found = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if found.0 {
            return found;
        }
        let _site = crate::hangwatch::at(&"waiting for the aside pass over the heap's buffer views (VICTORY_HEAP_PASS_ASIDE)");
        while !found.0 {
            found = self.1.wait(found).unwrap_or_else(|e| e.into_inner());
        }
        WAITED[0].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
        found
    }

    /// Until the work is done.
    pub(crate) fn wait(&self) {
        drop(self.done());
    }

    /// What was found (once).
    pub(crate) fn take(&self) -> Vec<Refresh> {
        self.done().1.take().unwrap_or_default()
    }

    fn found(&self, what: Vec<Refresh>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = (true, Some(what));
        self.1.notify_all();
    }
}

/// `VICTORY_HEAP_PASS_ASIDE` (default 1; 0: in the recording thread).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_ASIDE").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Per host heap (its address): the pass begun last, while it runs.
static PENDING: Mutex<Option<std::collections::HashMap<usize, std::sync::Arc<Coming>>>> = Mutex::new(None);

/// Until the pass begun last for host heap `host` is done (at once if none
/// runs).
pub(crate) fn wait_pass(host: usize) {
    let coming = PENDING.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&host).cloned());
    if let Some(coming) = coming {
        coming.wait();
    }
}

/// `work` begins on the thread for it, one after another (in order); here
/// and now if there is no such thread. `host`: the host heap it is for.
pub(crate) fn begin(host: usize, work: impl FnOnce() -> Vec<Refresh> + Send + 'static) -> std::sync::Arc<Coming> {
    static THREAD: OnceLock<Option<Mutex<std::sync::mpsc::Sender<Job>>>> = OnceLock::new();
    let coming = std::sync::Arc::new(Coming(Mutex::new((false, None)), std::sync::Condvar::new()));
    WAITED[1].fetch_add(1, Ordering::Relaxed);
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert(host, coming.clone());
    let told = coming.clone();
    let job: Job = Box::new(move || {
        // (However it ends, the list that waits goes on.)
        let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_default();
        told.found(found);
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = pending.as_mut()
            && m.get(&host).is_some_and(|c| std::sync::Arc::ptr_eq(c, &told))
        {
            m.remove(&host);
        }
    });
    let thread = THREAD.get_or_init(|| {
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("victory-aside".into())
            .spawn(move || jobs.into_iter().for_each(|job| job()))
            .ok()
            .map(|_| Mutex::new(send))
    });
    match thread {
        Some(send) => {
            if let Err(job) = send.lock().unwrap_or_else(|e| e.into_inner()).send(job) {
                (job.0)();
            }
        }
        None => job(),
    }
    coming
}

/// The refreshes `found` go into `copies` at `at`, as `defer_refresh` would
/// have put them there: one whose key was in the set at that time
/// (`refreshed`, the set as it was) is left out; a refresh pushed after
/// `at` whose key the set then held is left out in its turn, until a step
/// that empties the set (a title copy, a step that runs work).
pub(super) fn splice(
    copies: &mut Vec<Step>,
    set: &mut std::collections::HashSet<(usize, usize, usize)>,
    at: usize,
    mut refreshed: std::collections::HashSet<(usize, usize, usize)>,
    found: Vec<Refresh>,
) {
    let at = at.min(copies.len());
    let mut put: Vec<Step> = Vec::with_capacity(found.len());
    for (shadow, address, len) in found {
        let key = (std::sync::Arc::as_ptr(&shadow) as usize, address as usize, len as usize);
        if refreshed.insert(key) {
            put.push(Step::Refresh(shadow, address as usize, len as usize));
        }
    }
    if put.is_empty() {
        return;
    }
    let after: Vec<Step> = copies.drain(at..).collect();
    copies.extend(put);
    // Whether the set was emptied after `at`: then the set now is the
    // recording's own, untouched by what was put in.
    let mut emptied = false;
    for step in after {
        match &step {
            Step::Copy(..) | Step::Run(..) => {
                emptied = true;
                refreshed.clear();
            }
            Step::Refresh(shadow, address, len) => {
                let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                if !refreshed.insert(key) {
                    continue;
                }
            }
            _ => {}
        }
        copies.push(step);
    }
    // The set as `defer_refresh` would see it now.
    if !emptied {
        *set = refreshed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (P6) What `defer_refresh` would have left had the pass run inline:
    /// the pass's refreshes at their place, a later refresh of the same key
    /// left out until a title copy or a work step empties the set, one whose
    /// key the set held at the place left out.
    #[test]
    fn spliced_refreshes_stand_where_the_inline_pass_put_them() {
        // Steps are matched by kind and key alone: a stand-in host copy is
        // not needed, the keys are the copy's address. Real copies need a
        // device, so the splice is checked on keys through `Refresh` steps
        // built from one copy the shadow registry makes when there is a host.
        let Some(_host) = crate::host::host() else {
            return; // no host GPU here
        };
        const LEN: usize = 1 << 16;
        let title: &'static mut [u8] = Box::leak(vec![0u8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0a51usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start, 64).expect("a host copy");
        let s = start as usize;
        let key = |a: usize, n: usize| (std::sync::Arc::as_ptr(&shadow) as usize, a, n);
        let keys = |copies: &[Step]| -> Vec<String> {
            copies
                .iter()
                .map(|c| match c {
                    Step::Refresh(_, a, n) => format!("r{}+{}", a - s, n),
                    Step::Copy(..) => "copy".into(),
                    Step::Run(..) => "run".into(),
                    _ => "other".into(),
                })
                .collect()
        };
        // Before the place: refresh 0+16 (in the set). After it: refresh
        // 32+16 (also found by the pass: left out), a title copy, refresh
        // 32+16 again (kept: the copy emptied the set).
        let mut copies = vec![Step::Refresh(shadow.clone(), s, 16)];
        let mut set: std::collections::HashSet<(usize, usize, usize)> = [key(s, 16)].into_iter().collect();
        let at = copies.len();
        let refreshed = set.clone();
        copies.push(Step::Refresh(shadow.clone(), s + 32, 16));
        set.insert(key(s + 32, 16));
        copies.push(Step::Copy(0, 0, 0));
        set.clear();
        copies.push(Step::Refresh(shadow.clone(), s + 32, 16));
        set.insert(key(s + 32, 16));
        let found = vec![(shadow.clone(), start, 16), (shadow.clone(), start + 32, 16), (shadow.clone(), start + 64, 8)];
        splice(&mut copies, &mut set, at, refreshed, found);
        assert_eq!(keys(&copies), ["r0+16", "r32+16", "r64+8", "copy", "r32+16"]);
        // The set after a copy is the recording's own.
        assert!(set.contains(&key(s + 32, 16)) && !set.contains(&key(s + 64, 8)));
        // Without a copy after the place, the set takes the pass's keys.
        let mut copies = vec![Step::Refresh(shadow.clone(), s + 128, 16)];
        let mut set: std::collections::HashSet<(usize, usize, usize)> = [key(s + 128, 16)].into_iter().collect();
        splice(&mut copies, &mut set, 0, Default::default(), vec![(shadow.clone(), start + 128, 16), (shadow.clone(), start + 256, 16)]);
        assert_eq!(keys(&copies), ["r128+16", "r256+16"]);
        assert!(set.contains(&key(s + 256, 16)));
        crate::shadow::unregister(resource, start);
    }

    #[test]
    fn a_pass_is_waited_for_and_a_panic_lets_the_list_go_on() {
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let coming = begin(0x51, move || {
            let _ = wait.recv();
            Vec::new()
        });
        assert!(!coming.0.lock().unwrap().0);
        go.send(()).unwrap();
        wait_pass(0x51);
        assert!(coming.take().is_empty());
        assert!(begin(0x52, || panic!("a panic of the work")).take().is_empty());
        wait_pass(0x52);
    }
}
