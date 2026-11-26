//! (P6, port of fxceless 895a23b, `VICTORY_HEAP_PASS_MARKED`; default 0, the
//! author's) Which slots the pass over the view heap's buffer views
//! (`draw::heap_buffer_pass`) looks at again. Unset, it looks at the slots
//! whose buffer view was written or went since the last pass, and at every
//! buffer slot every 16 passes (84,027 slots, 178 ms a pass in the author's
//! run 286). Set, a slot is looked at again only when
//! - its buffer view was written or went (the heap's own set of changes);
//! - the view table entry its record names was freed (the title released
//!   the resource: `descriptor::forget`), which leaves the record as it was;
//! - its record in title memory (the first 12 bytes: tag, entry,
//!   generation) is not the one seen at the last look (a record copied by
//!   the title itself, past the API);
//! - it found no host copy at its last look: every 16 passes, as before;
//! - every [`FULL_EVERY`] passes, all of them, as a safety net.
//!
//! Exactness: the full pass every 16 exists because a view goes stale
//! without a write (its resource released: the entry is freed and named
//! here) and because a slot written before its memory had a host copy is
//! made only later (still looked at every 16 passes). A record changed
//! behind the API is found by the compare. What is not covered: a host copy
//! that is let go and made again for the same range while the slot keeps
//! its record and view (the pass keeps the old copy for it until the next
//! full pass); hence not the default.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Every this many passes, every slot (as the full pass of before, rarer).
pub(crate) const FULL_EVERY: u64 = 256;

/// `VICTORY_HEAP_PASS_MARKED=1` (0 default, as the author's).
pub(crate) fn marked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_MARKED").is_some_and(|v| v == "1"))
}

/// Table entries freed since a pass last asked ([`slots`]); kept only once
/// one asked.
static FORGOTTEN: Mutex<(bool, Vec<u32>)> = Mutex::new((false, Vec::new()));

/// `descriptor::forget` freed view table entry `i`.
pub(crate) fn forgotten(i: u32) {
    if !marked() {
        return;
    }
    let mut f = FORGOTTEN.lock().unwrap_or_else(|e| e.into_inner());
    if f.0 {
        f.1.push(i);
    }
}

/// Per host heap: the records (tag, entry, generation) of the buffer slots
/// as last looked at.
static SEEN: Mutex<Option<HashMap<usize, HashMap<u32, [u32; 3]>>>> = Mutex::new(None);

/// The slots of the pass number `pass` over host heap `host`, and whether
/// it is a full one (the caller forgets what it kept first). `changed`: the
/// heap's own changes; `had_copy`: slots that found a host copy at their
/// last look; `all`: every buffer slot; `record`: a slot's record words as
/// they are in title memory now (None: unreadable).
pub(crate) fn slots(
    host: usize,
    pass: u64,
    changed: Vec<u32>,
    had_copy: &[u32],
    all: impl FnOnce() -> Vec<u32>,
    record: impl Fn(u32) -> Option<[u32; 3]>,
) -> (Vec<u32>, bool) {
    let freed: HashSet<u32> = {
        let mut f = FORGOTTEN.lock().unwrap_or_else(|e| e.into_inner());
        f.0 = true;
        std::mem::take(&mut f.1).into_iter().collect()
    };
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default).entry(host).or_default();
    let every = all();
    if pass.is_multiple_of(FULL_EVERY) {
        seen.clear();
        for &s in &every {
            if let Some(r) = record(s) {
                seen.insert(s, r);
            }
        }
        return (every, true);
    }
    let had: HashSet<u32> = had_copy.iter().copied().collect();
    let mut out: HashSet<u32> = changed.into_iter().collect();
    for &s in &every {
        let now = record(s);
        let was = seen.get(&s).copied();
        let differs = now != was;
        let named_freed = was.is_some_and(|r| freed.contains(&r[1]));
        let no_copy = pass.is_multiple_of(16) && !had.contains(&s);
        if differs || named_freed || no_copy {
            out.insert(s);
        }
        match now {
            Some(r) => {
                seen.insert(s, r);
            }
            None => {
                seen.remove(&s);
            }
        }
    }
    let mut out: Vec<u32> = out.into_iter().collect();
    out.sort_unstable();
    (out, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(host: usize, pass: u64, changed: Vec<u32>, had: &[u32], records: &HashMap<u32, [u32; 3]>) -> (Vec<u32>, bool) {
        let all: Vec<u32> = {
            let mut a: Vec<u32> = records.keys().copied().collect();
            a.sort_unstable();
            a
        };
        slots(host, pass, changed, had, || all, |s| records.get(&s).copied())
    }

    #[test]
    fn only_marked_slots_between_full_passes() {
        // A heap address no other test uses.
        let host = 0x5eed_0001;
        let mut records: HashMap<u32, [u32; 3]> = (0..100).map(|s| (s, [1, s + 1000, 1])).collect();
        let had: Vec<u32> = (0..100).collect();
        let (first, full) = run(host, 0, vec![], &had, &records);
        assert!(full);
        assert_eq!(first.len(), 100);
        // Nothing changed: nothing looked at (pass 1, not a 16th).
        assert_eq!(run(host, 1, vec![], &had, &records), (vec![], false));
        // A write through the API.
        assert_eq!(run(host, 2, vec![7], &had, &records).0, vec![7]);
        // A record copied as bytes.
        records.insert(9, [1, 5555, 3]);
        assert_eq!(run(host, 3, vec![], &had, &records).0, vec![9]);
        assert_eq!(run(host, 4, vec![], &had, &records).0, Vec::<u32>::new());
        // The entry a record names freed.
        let mut f = super::FORGOTTEN.lock().unwrap();
        f.0 = true;
        f.1.push(1020);
        drop(f);
        assert_eq!(run(host, 5, vec![], &had, &records).0, vec![20]);
        // Every 16th: the slots that found no copy.
        let had_some: Vec<u32> = (0..100).filter(|&s| s != 33).collect();
        assert_eq!(run(host, 16, vec![], &had_some, &records).0, vec![33]);
        // Full again.
        assert!(run(host, FULL_EVERY, vec![], &had, &records).1);
    }
}
