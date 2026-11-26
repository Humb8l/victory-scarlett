//! (F21 meter, with `VICTORY_CHUNK_METER=1` on the hle side) The title's
//! arena segment C: chunks of per-draw data, each written once by one
//! recording context and final when its list is closed (0x1424749d0 hands
//! the retired chunks with the list's fence token), reused only once that
//! token has passed. Measured here: the bytes of the chunks retired against
//! the bytes our refreshes take from the segment, and the hazard the analyst
//! names: a chunk retired again (so reused) with no refresh of ours over it
//! since it was last retired ("chunk reused before read").

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The segment: start, bytes, chunk bytes (0: not known yet).
static SEGMENT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Per chunk (by index in the segment): the number of its last retirement,
/// and of the last refresh of ours over it.
static CHUNKS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static SEQ: AtomicU64 = AtomicU64::new(1);
/// Since the last line: chunks retired, their bytes, retired chunks outside
/// the segment, chunks retired again with no refresh over them since their
/// last retirement, bytes our refreshes took from the segment, refreshes
/// that touched it.
static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The segment as the title's globals give it now.
pub fn segment(start: u64, len: u64, chunk: u64) {
    SEGMENT[0].store(start, Ordering::Relaxed);
    SEGMENT[1].store(len, Ordering::Relaxed);
    SEGMENT[2].store(chunk, Ordering::Relaxed);
}

fn index(address: u64) -> Option<usize> {
    let (start, len, chunk) = (SEGMENT[0].load(Ordering::Relaxed), SEGMENT[1].load(Ordering::Relaxed), SEGMENT[2].load(Ordering::Relaxed));
    (chunk != 0 && address >= start && address < start + len).then(|| ((address - start) / chunk) as usize)
}

/// A list was closed: `chunks` (their addresses) are retired with its token.
pub fn retired(chunks: &[u64]) {
    let chunk = SEGMENT[2].load(Ordering::Relaxed);
    let mut all = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    for &address in chunks {
        let Some(i) = index(address) else {
            COUNTS[2].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if all.len() <= i {
            all.resize(i + 1, (0, 0));
        }
        let (last_retired, last_read) = all[i];
        if last_retired != 0 && last_read < last_retired {
            COUNTS[3].fetch_add(1, Ordering::Relaxed);
        }
        all[i].0 = SEQ.fetch_add(1, Ordering::Relaxed);
        COUNTS[0].fetch_add(1, Ordering::Relaxed);
        COUNTS[1].fetch_add(chunk, Ordering::Relaxed);
    }
}

/// A refresh of ours took title memory `[from, to)` into a host copy.
pub(crate) fn refreshed(from: usize, to: usize) {
    let (start, len) = (SEGMENT[0].load(Ordering::Relaxed), SEGMENT[1].load(Ordering::Relaxed));
    if len == 0 {
        return;
    }
    let (a, b) = ((from as u64).max(start), (to as u64).min(start + len));
    if a >= b {
        return;
    }
    COUNTS[4].fetch_add(b - a, Ordering::Relaxed);
    COUNTS[5].fetch_add(1, Ordering::Relaxed);
    let (Some(first), Some(last)) = (index(a), index(b - 1)) else {
        return;
    };
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut all = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    if all.len() <= last {
        all.resize(last + 1, (0, 0));
    }
    for c in &mut all[first..=last] {
        c.1 = seq;
    }
}

/// The counts since the last call: chunks retired, their bytes, outside the
/// segment, reused before read, bytes refreshed from the segment, refreshes.
pub fn take() -> [u64; 6] {
    [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Ordering::Relaxed))
}
