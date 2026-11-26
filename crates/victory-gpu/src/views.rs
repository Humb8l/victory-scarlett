//! (P6, port of avatarmanz 6c17ce9 `land::each_view`) Threads kept for
//! making descriptors: the heaps' records copied as bytes
//! (`descriptor::sync_heaps`) and the pass over the buffer views shaders pick
//! by index (`draw::State::heap_buffers`). `VICTORY_VIEW_THREADS` (5 with the
//! caller, the author's; 1: on the caller alone, as before).
//!
//! A pool of its own, not the wide workers of `crate::shadow`: the heap sync
//! runs beside the submitting thread's preparing (`VICTORY_HEAP_SYNC_ASIDE`),
//! which keeps those busy.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Each = dyn Fn(usize) + Sync;

/// One call of [`each`]: what to call, the next number, the numbers done,
/// and the first panic.
struct Round {
    /// Lifetime erased: [`each`] does not return before every number is done.
    f: &'static Each,
    count: usize,
    next: AtomicUsize,
    done: Mutex<usize>,
    over: Condvar,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl Round {
    /// Takes numbers until none is left.
    fn work(&self) {
        loop {
            let n = self.next.fetch_add(1, Ordering::Relaxed);
            if n >= self.count {
                return;
            }
            if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.f)(n))) {
                self.panic.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(e);
            }
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            *done += 1;
            if *done == self.count {
                self.over.notify_all();
            }
        }
    }
}

struct Pool {
    rounds: Mutex<std::collections::VecDeque<Arc<Round>>>,
    more: Condvar,
    threads: usize,
}

/// `VICTORY_VIEW_THREADS`: threads with the caller (1-16; 5).
pub(crate) fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_VIEW_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(5)
            .clamp(1, 16)
    })
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            rounds: Mutex::new(std::collections::VecDeque::new()),
            more: Condvar::new(),
            threads: threads(),
        }));
        for i in 1..pool.threads {
            let _ = std::thread::Builder::new().name(format!("victory-views-{i}")).spawn(move || {
                loop {
                    let round = {
                        let mut rounds = pool.rounds.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            // A round whose numbers are all taken leaves the line.
                            while rounds.front().is_some_and(|r| r.next.load(Ordering::Relaxed) >= r.count) {
                                rounds.pop_front();
                            }
                            if let Some(r) = rounds.front() {
                                break r.clone();
                            }
                            rounds = pool.more.wait(rounds).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    round.work();
                }
            });
        }
        pool
    })
}

/// Calls `f` with every number below `count`, once each, on this thread and
/// on the kept threads; returns when all are done. In order on this thread
/// alone when `count` is under `serial_below` or with one thread. A panic of
/// `f` reaches the caller after every number was done.
pub(crate) fn each(count: usize, serial_below: usize, f: &(dyn Fn(usize) + Sync)) {
    if count == 0 {
        return;
    }
    if threads() <= 1 || count < serial_below.max(2) {
        (0..count).for_each(f);
        return;
    }
    let pool = pool();
    // SAFETY: the round is not left before `done == count`: no call of `f`
    // runs after this function returns (the kept threads only take numbers
    // below `count`, and each is counted done after its call).
    let f: &'static Each = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static Each>(f) };
    let round = Arc::new(Round {
        f,
        count,
        next: AtomicUsize::new(0),
        done: Mutex::new(0),
        over: Condvar::new(),
        panic: Mutex::new(None),
    });
    pool.rounds.lock().unwrap_or_else(|e| e.into_inner()).push_back(round.clone());
    pool.more.notify_all();
    round.work();
    let mut done = round.done.lock().unwrap_or_else(|e| e.into_inner());
    while *done < count {
        done = round.over.wait(done).unwrap_or_else(|e| e.into_inner());
    }
    drop(done);
    if let Some(e) = round.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
        std::panic::resume_unwind(e);
    }
}

#[cfg(test)]
mod tests {
    use super::each;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn every_number_once_from_several_callers() {
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for count in [0usize, 1, 2, 7, 100, 1000] {
                        let seen: Vec<AtomicU32> = (0..count).map(|_| AtomicU32::new(0)).collect();
                        each(count, 2, &|n| {
                            seen[n].fetch_add(1, Ordering::Relaxed);
                        });
                        assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
                    }
                });
            }
        });
    }

    #[test]
    fn a_panic_reaches_the_caller_after_the_rest() {
        let seen = AtomicU32::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            each(64, 2, &|n| {
                seen.fetch_add(1, Ordering::Relaxed);
                assert!(n != 10, "the tenth");
            })
        }));
        assert!(r.is_err());
        assert_eq!(seen.load(Ordering::Relaxed), 64);
        let after = AtomicU32::new(0);
        each(64, 2, &|_| {
            after.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(after.load(Ordering::Relaxed), 64);
    }
}
